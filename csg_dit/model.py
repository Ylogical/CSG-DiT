import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


def broadcat(tensors, dim = -1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim = dim)


def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(self, dim, pt_seq_len, theta=10000):
        super().__init__()
        freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        t = torch.arange(pt_seq_len).float()

        freqs = torch.einsum('..., f -> ... f', t, freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim = -1)

        self.register_buffer('freqs_cos', freqs.cos().view(-1, freqs.shape[-1]))
        self.register_buffer('freqs_sin', freqs.sin().view(-1, freqs.shape[-1]))

    def forward(self, t):
        return t * self.freqs_cos + rotate_half(t) * self.freqs_sin


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])

    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


def scaled_dot_product_attention(query, key, value, dropout_p=0.0) -> torch.Tensor:
    return F.scaled_dot_product_attention(query, key, value, dropout_p=dropout_p)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.q_norm(q)
        k = self.k_norm(k)

        if rope is not None:
            q = rope(q)
            k = rope(k)

        x = scaled_dot_product_attention(q, k, v,
                                          dropout_p=self.attn_drop.p if self.training else 0.)

        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop=0.0,
        bias=True
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class MultiScaleEncoder(nn.Module):
    def __init__(self, in_chans=3, hidden_size=768, patch_size=16, input_size=256):
        super().__init__()
        self.patch_size = patch_size
        self.input_size = input_size
        D = hidden_size

        self.enc1 = ConvBlock(in_chans, 64)
        self.down1 = nn.Conv2d(64, 128, 2, 2)

        self.enc2 = ConvBlock(128, 256)
        self.down2 = nn.Conv2d(256, 512, 2, 2)

        self.enc3 = ConvBlock(512, D)

        stride = patch_size // 4
        self.to_tokens = nn.Sequential(
            nn.AvgPool2d(kernel_size=stride, stride=stride),
            nn.Conv2d(D, D, kernel_size=1),
        )

        self.skip_channels = [64, 256, D]
        self.num_patches = (input_size // patch_size) ** 2

    def forward(self, x):
        f1 = self.enc1(x)
        f2 = self.enc2(self.down1(f1))
        f3 = self.enc3(self.down2(f2))
        tokens_2d = self.to_tokens(f3)
        tokens = tokens_2d.flatten(2).transpose(1, 2)
        return tokens, [f1, f2, f3]


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)
        return self.conv(torch.cat([x, skip], dim=1))


class UBAM(nn.Module):
    def __init__(self, skip_ch, cond_dim, t_proj_dim=64, gate_hidden=16):
        super().__init__()
        self.t_mlp = nn.Sequential(
            nn.Linear(cond_dim, t_proj_dim), nn.SiLU(),
            nn.Linear(t_proj_dim, 2 * skip_ch),
        )
        self.t_gate_proj = nn.Linear(cond_dim, gate_hidden)
        self.gate_conv = nn.Sequential(
            nn.Conv2d(2 + gate_hidden, gate_hidden, 3, padding=1),
            nn.SiLU(), nn.Conv2d(gate_hidden, 1, 3, padding=1),
        )
        self.alpha = nn.Parameter(torch.ones(1))
        nn.init.zeros_(self.gate_conv[-1].weight)
        nn.init.zeros_(self.gate_conv[-1].bias)

    def forward(self, skip, e_t, uncertainty, boundary):
        gamma, beta = self.t_mlp(e_t).chunk(2, dim=1)
        out = skip * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        size = skip.shape[-2:]
        u = F.interpolate(uncertainty, size=size, mode='bilinear', align_corners=False)
        b = F.interpolate(boundary, size=size, mode='bilinear', align_corners=False)
        time = self.t_gate_proj(e_t)[:, :, None, None].expand(-1, -1, *size)
        gate = 1.0 + self.alpha * self.gate_conv(torch.cat([u, b, time], dim=1)).tanh()
        return out * gate


class HierarchicalConditionEncoder(nn.Module):
    def __init__(self, in_chans, hidden_size, patch_size, input_size,
                 dino_repo=None, dino_weights=None, dino_pretrained=True):
        super().__init__()
        self.cnn = MultiScaleEncoder(in_chans, hidden_size, patch_size, input_size)
        self.skip_channels = self.cnn.skip_channels
        self.grid = input_size // patch_size
        self.dino = torch.hub.load(
            dino_repo or 'facebookresearch/dinov2:main', 'dinov2_vitb14',
            source='local' if dino_repo else 'github',
            pretrained=dino_pretrained and dino_weights is None,
        )
        if dino_pretrained and dino_weights:
            state = torch.load(dino_weights, map_location='cpu', weights_only=True)
            self.dino.load_state_dict(state, strict=True)
        self.dino.requires_grad_(False)
        self.dino.eval()
        self.cnn_norm = nn.LayerNorm(hidden_size)
        self.dino_norm = nn.LayerNorm(768)
        self.to_tokens = nn.Linear(hidden_size + 768, hidden_size)
        self.register_buffer('image_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
                             persistent=False)
        self.register_buffer('image_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
                             persistent=False)

    def train(self, mode=True):
        super().train(mode)
        self.dino.eval()
        return self

    def forward(self, image):
        cnn_tokens, skips = self.cnn(image)
        rgb = (image + 1.0) * 0.5
        if rgb.shape[1] == 1:
            rgb = rgb.expand(-1, 3, -1, -1)
        rgb = (rgb - self.image_mean) / self.image_std
        with torch.no_grad():
            rgb = F.interpolate(rgb, size=(self.grid * 14, self.grid * 14),
                                mode='bicubic', align_corners=False, antialias=True)
            vit_tokens = self.dino.forward_features(rgb)['x_norm_patchtokens']
        tokens = self.to_tokens(torch.cat(
            [self.cnn_norm(cnn_tokens), self.dino_norm(vit_tokens)], dim=-1))
        return tokens, skips


class EnhancedFinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.refinement = nn.Sequential(
            RMSNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.norm_final = RMSNorm(hidden_size)
        intermediate_dim = hidden_size // 2
        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(hidden_size, intermediate_dim, bias=True),
            nn.GELU(),
        )
        self.output_proj = nn.Linear(intermediate_dim, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        x = x + self.refinement(x)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.output_proj(self.mlp_head(x))


class UNetDecoder(nn.Module):
    def __init__(self, hidden_size, out_channels, mask_skip_channels, image_skip_channels,
                 patch_size=16):
        super().__init__()
        D = hidden_size

        self.norm   = RMSNorm(D)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(D, 2 * D, bias=True)
        )

        self.stage0 = nn.Sequential(
            nn.Upsample(scale_factor=patch_size / 8, mode='bilinear', align_corners=True),
            ConvBlock(D, D // 2),
        )

        sc = [mask_skip_channels[i] + image_skip_channels[i] for i in range(3)]
        self.ubam3 = UBAM(mask_skip_channels[2], cond_dim=D)
        self.ubam2 = UBAM(mask_skip_channels[1], cond_dim=D)
        self.ubam1 = UBAM(mask_skip_channels[0], cond_dim=D)

        self.up1 = UpBlock(D // 2, sc[2], D // 4)
        self.up2 = UpBlock(D // 4, sc[1], D // 8)
        self.up3 = UpBlock(D // 8, sc[0], 64)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x, c, mask_skips, image_skips, t_emb, uncertainty, boundary):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)

        B, N, D = x.shape
        hw = math.isqrt(N)
        assert hw * hw == N, f"number of mask tokens {N} is not a perfect square"
        x = x.reshape(B, hw, hw, D).permute(0, 3, 1, 2)

        x = self.stage0(x)

        def _joint(idx, ubam):
            state = ubam(mask_skips[idx], t_emb, uncertainty, boundary)
            return torch.cat([state, image_skips[idx]], dim=1)

        x = self.up1(x, _joint(2, self.ubam3))
        x = self.up2(x, _joint(1, self.ubam2))
        x = self.up3(x, _joint(0, self.ubam1))

        return self.out_conv(x)


class DFCA(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, tau_min: float = 0.5, tau_max: float = 1.5,
                 attn_drop: float = 0.0, kv_mode: str = 'split'):
        super().__init__()
        assert hidden_size % num_heads == 0
        if kv_mode not in ('split', 'v_only'):
            raise ValueError(f"kv_mode must be 'split' or 'v_only', got {kv_mode!r}")
        self.num_heads = num_heads
        self.head_dim  = hidden_size // num_heads
        self.attn_drop = attn_drop
        self.kv_mode   = kv_mode
        self.q_norm = RMSNorm(self.head_dim)
        if kv_mode == 'v_only':
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.k_norm_sem = RMSNorm(self.head_dim)
            self.k_norm_det = RMSNorm(self.head_dim)
        self.log_tau_sem = nn.Parameter(torch.full((num_heads,), math.log(tau_max)))
        self.log_tau_det = nn.Parameter(torch.full((num_heads,), math.log(tau_min)))

        self.band_low_log = nn.Parameter(torch.tensor(math.log(8.0)))
        init_gap = 1.0 / 1.5 - 1.0 / 8.0
        self.band_gap_raw = nn.Parameter(torch.tensor(math.log(math.expm1(init_gap))))

        self.norm_q  = RMSNorm(hidden_size)
        self.norm_kv = RMSNorm(hidden_size)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.to_q = nn.Linear(hidden_size, hidden_size, bias=False)

        if kv_mode == 'v_only':
            self.to_k     = nn.Linear(hidden_size, hidden_size, bias=False)
            self.to_v_sem = nn.Linear(hidden_size, hidden_size, bias=False)
            self.to_v_det = nn.Linear(hidden_size, hidden_size, bias=False)
        else:
            self.to_k_sem = nn.Linear(hidden_size, hidden_size, bias=False)
            self.to_v_sem = nn.Linear(hidden_size, hidden_size, bias=False)
            self.to_k_det = nn.Linear(hidden_size, hidden_size, bias=False)
            self.to_v_det = nn.Linear(hidden_size, hidden_size, bias=False)

        self.detail_enhance = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, 3, 1, 1, groups=hidden_size, bias=False),
            nn.GroupNorm(min(32, hidden_size), hidden_size),
            nn.GELU(),
        )

        self.proj_alpha = nn.Linear(hidden_size, 1, bias=True)

        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def _freq_decompose(self, kv_in: torch.Tensor, hw: int):
        B, Ni, D = kv_in.shape
        feat     = kv_in.reshape(B, hw, hw, D).permute(0, 3, 1, 2).contiguous()

        with torch.amp.autocast(device_type=feat.device.type, enabled=False):
            feat_f   = feat.float()
            feat_fft = torch.fft.fft2(feat_f, dim=(-2, -1), norm='ortho')
            feat_fft = torch.fft.fftshift(feat_fft, dim=(-2, -1))

            coords = torch.arange(hw, device=feat.device, dtype=torch.float32) - hw // 2
            yy, xx = torch.meshgrid(coords, coords, indexing='ij')
            r2     = xx ** 2 + yy ** 2

            sigma_s = (hw / torch.exp(self.band_low_log)).clamp(min=1.0)
            sigma_n = sigma_s + hw * F.softplus(self.band_gap_raw)
            m_low   = torch.exp(-r2 / (2 * sigma_s ** 2))
            m_high  = (1.0 - m_low) * torch.exp(-r2 / (2 * sigma_n ** 2))

            m_low    = m_low.unsqueeze(0).unsqueeze(0)
            m_high   = m_high.unsqueeze(0).unsqueeze(0)

            fft_low  = feat_fft * m_low
            fft_high = feat_fft * m_high

            kv_low_spatial  = torch.fft.ifft2(torch.fft.ifftshift(fft_low,  dim=(-2, -1)),
                                              dim=(-2, -1), norm='ortho').real
            kv_high_spatial = torch.fft.ifft2(torch.fft.ifftshift(fft_high, dim=(-2, -1)),
                                              dim=(-2, -1), norm='ortho').real

        kv_low_spatial  = kv_low_spatial.to(feat.dtype)
        kv_high_spatial = self.detail_enhance(kv_high_spatial.to(feat.dtype))

        kv_low  = kv_low_spatial.permute(0, 2, 3, 1).reshape(B, Ni, D)
        kv_high = kv_high_spatial.permute(0, 2, 3, 1).reshape(B, Ni, D)
        return kv_low, kv_high

    def forward(
        self,
        mask_tokens:  torch.Tensor,
        image_tokens: torch.Tensor,
        c:            torch.Tensor,
        t_emb:        torch.Tensor,
        rope=None,
    ) -> torch.Tensor:
        B, N,  D  = mask_tokens.shape
        _,  Ni, _ = image_tokens.shape
        H,  Hd    = self.num_heads, self.head_dim

        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        q_in  = modulate(self.norm_q(mask_tokens),  shift, scale)
        kv_in = self.norm_kv(image_tokens)

        kv_hw = math.isqrt(Ni)
        assert kv_hw * kv_hw == Ni, (
            f"number of image tokens {Ni} is not a perfect square, a square "
            f"feature map is required; check the image_encoder output"
        )
        kv_low, kv_high = self._freq_decompose(kv_in, kv_hw)

        q     = self.to_q(q_in).reshape(B, N, H, Hd).transpose(1, 2)
        v_sem = self.to_v_sem(kv_low).reshape(B, Ni, H, Hd).transpose(1, 2)
        v_det = self.to_v_det(kv_high).reshape(B, Ni, H, Hd).transpose(1, 2)
        q = self.q_norm(q)
        if self.kv_mode == 'v_only':
            k = self.k_norm(self.to_k(kv_in).reshape(B, Ni, H, Hd).transpose(1, 2))
            if rope is not None:
                q, k = rope(q), rope(k)
            k_sem = k_det = k
        else:
            k_sem = self.k_norm_sem(self.to_k_sem(kv_low).reshape(B, Ni, H, Hd).transpose(1, 2))
            k_det = self.k_norm_det(self.to_k_det(kv_high).reshape(B, Ni, H, Hd).transpose(1, 2))
            if rope is not None:
                q, k_sem, k_det = rope(q), rope(k_sem), rope(k_det)

        alpha = torch.sigmoid(self.proj_alpha(t_emb))

        scale_sem = (1.0 / (math.sqrt(Hd) * self.log_tau_sem.float().exp())).view(1, H, 1, 1)
        scale_det = (1.0 / (math.sqrt(Hd) * self.log_tau_det.float().exp())).view(1, H, 1, 1)

        with torch.amp.autocast(device_type=mask_tokens.device.type, enabled=False):
            q_f, k_sem_f, v_sem_f = q.float(), k_sem.float(), v_sem.float()
            k_det_f, v_det_f      = k_det.float(), v_det.float()
            alpha_f               = alpha.float().unsqueeze(1).unsqueeze(-1)

            attn_sem = ((q_f @ k_sem_f.transpose(-2, -1)) * scale_sem).softmax(dim=-1)
            if self.training and self.attn_drop > 0.0:
                attn_sem = F.dropout(attn_sem, p=self.attn_drop, training=True)
            out_sem = attn_sem @ v_sem_f

            attn_det = ((q_f @ k_det_f.transpose(-2, -1)) * scale_det).softmax(dim=-1)
            if self.training and self.attn_drop > 0.0:
                attn_det = F.dropout(attn_det, p=self.attn_drop, training=True)
            out_det = attn_det @ v_det_f

            out = alpha_f * out_sem + (1.0 - alpha_f) * out_det

        out = out.to(mask_tokens.dtype).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class CSGDiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, feat_rope=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class CSGDiT_Segmentation(nn.Module):
    def __init__(
        self,
        input_size=256,
        patch_size=16,
        mask_channels=1,
        image_channels=3,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.1,
        proj_drop=0.1,
        dfca_attn_drop=0.1,
        dfca_kv_mode='split',
        dino_repo=None,
        dino_weights=None,
        dino_pretrained=True,
        x0_softmax=None,
    ):
        super().__init__()
        self.mask_channels = mask_channels
        if x0_softmax is None:
            x0_softmax = (mask_channels > 1)
        if x0_softmax and mask_channels == 1:
            raise ValueError('x0_softmax needs mask_channels > 1; a single-channel '
                             'softmax is constant 1, so binary masks use the tanh head')
        self.x0_softmax = x0_softmax
        if image_channels not in (1, 3):
            raise ValueError('HCSE supports one-channel or RGB condition images')
        if input_size % patch_size or patch_size % 4:
            raise ValueError('input_size must be divisible by patch_size, and patch_size by 4')
        self.register_buffer('sobel', torch.tensor([
            [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
            [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
        ]).unsqueeze(1))
        self.image_channels = image_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.t_embedder = TimestepEmbedder(hidden_size)

        self.mask_encoder = MultiScaleEncoder(
            in_chans=mask_channels, hidden_size=hidden_size,
            patch_size=patch_size, input_size=input_size
        )
        self.image_encoder = HierarchicalConditionEncoder(
            in_chans=image_channels, hidden_size=hidden_size,
            patch_size=patch_size, input_size=input_size,
            dino_repo=dino_repo, dino_weights=dino_weights, dino_pretrained=dino_pretrained,
        )

        self.modality_embed_image = nn.Parameter(torch.zeros(1, 1, hidden_size), requires_grad=True)
        self.modality_embed_mask  = nn.Parameter(torch.zeros(1, 1, hidden_size), requires_grad=True)
        torch.nn.init.normal_(self.modality_embed_image, std=0.02)
        torch.nn.init.normal_(self.modality_embed_mask,  std=0.02)

        num_patches = self.mask_encoder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=hidden_size // num_heads // 2,
            pt_seq_len=input_size // patch_size,
        )

        self.pool_weight = nn.Linear(hidden_size, 1)
        self.cond_mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

        self.blocks = nn.ModuleList([
            CSGDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                        attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                        proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0)
            for i in range(depth)
        ])

        self.dfca_layers = nn.ModuleList([
            DFCA(hidden_size, num_heads, attn_drop=dfca_attn_drop, kv_mode=dfca_kv_mode)
            for _ in range(depth)
        ])

        self.final_layer = UNetDecoder(
            hidden_size=hidden_size,
            out_channels=mask_channels,
            mask_skip_channels=self.mask_encoder.skip_channels,
            image_skip_channels=self.image_encoder.skip_channels,
            patch_size=patch_size,
        )

        self.aux_token_head = EnhancedFinalLayer(hidden_size, patch_size, mask_channels)
        self._aux_out = None

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        for name, module in self.named_modules():
            if not name.startswith('image_encoder.dino'):
                _basic_init(module)

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.pos_embed.shape[1] ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        for encoder in (self.image_encoder.cnn, self.mask_encoder):
            for m in encoder.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.GroupNorm):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        nn.init.xavier_uniform_(self.cond_mlp[0].weight)
        nn.init.constant_(self.cond_mlp[0].bias, 0)
        nn.init.normal_(self.cond_mlp[2].weight, std=0.02)
        nn.init.constant_(self.cond_mlp[2].bias, 0)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        nn.init.trunc_normal_(self.final_layer.out_conv.weight, std=0.02)
        nn.init.constant_(self.final_layer.out_conv.bias, 0)

        for ubam in (self.final_layer.ubam1, self.final_layer.ubam2, self.final_layer.ubam3):
            nn.init.normal_(ubam.t_mlp[-1].weight, std=0.02)
            nn.init.zeros_(ubam.t_mlp[-1].bias)

        nn.init.constant_(self.aux_token_head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.aux_token_head.adaLN_modulation[-1].bias, 0)
        nn.init.trunc_normal_(self.aux_token_head.output_proj.weight, std=0.02)
        nn.init.constant_(self.aux_token_head.output_proj.bias, 0)
        nn.init.constant_(self.aux_token_head.refinement[-1].weight, 0)
        nn.init.constant_(self.aux_token_head.refinement[-1].bias, 0)

        for dfca in self.dfca_layers:
            nn.init.zeros_(dfca.proj.weight)
            nn.init.zeros_(dfca.proj.bias)
            nn.init.constant_(dfca.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(dfca.adaLN_modulation[-1].bias, 0)
            nn.init.zeros_(dfca.proj_alpha.weight)
            nn.init.zeros_(dfca.proj_alpha.bias)

    def encode_condition(self, image_cond):
        image_tokens, image_skips = self.image_encoder(image_cond)
        image_tokens = image_tokens + self.pos_embed + self.modality_embed_image
        weights = self.pool_weight(image_tokens).softmax(dim=1)
        context = (weights * image_tokens).sum(dim=1)
        with torch.amp.autocast(device_type=image_cond.device.type, enabled=False):
            gray = image_cond.float().mean(dim=1, keepdim=True)
            grad = F.conv2d(gray, self.sobel.float(), padding=1)
            magnitude = torch.sqrt(grad.pow(2).sum(dim=1, keepdim=True) + 1e-6)
            boundary = magnitude / (magnitude.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        return image_tokens, image_skips, context, boundary

    def compute_uncertainty(self, noisy_mask):
        with torch.amp.autocast(device_type=noisy_mask.device.type, enabled=False):
            return 1.0 - noisy_mask.float().mean(dim=1, keepdim=True).tanh().abs()

    def forward(self, x_mask_noisy, t, image_cond, condition=None):
        if condition is None:
            condition = self.encode_condition(image_cond)
        image_tokens, image_skips, context, boundary = condition

        t_emb = self.t_embedder(t)
        c = self.cond_mlp(torch.cat([t_emb, context], dim=-1))

        mask_tokens, mask_skips = self.mask_encoder(x_mask_noisy)
        mask_tokens = mask_tokens + self.pos_embed + self.modality_embed_mask

        for block, dfca in zip(self.blocks, self.dfca_layers):
            mask_tokens = block(mask_tokens, c, self.feat_rope)
            mask_tokens = mask_tokens + dfca(mask_tokens, image_tokens, c, t_emb,
                                             rope=self.feat_rope)

        if self.training:
            self._aux_out = self._x0_act(self.unpatchify(self.aux_token_head(mask_tokens, c)))

        uncertainty = self.compute_uncertainty(x_mask_noisy)
        return self._x0_act(self.final_layer(
            mask_tokens, c, mask_skips, image_skips, t_emb, uncertainty, boundary))

    def unpatchify(self, x):
        p, C = self.patch_size, self.mask_channels
        h = math.isqrt(x.shape[1])
        x = x.reshape(x.shape[0], h, h, p, p, C)
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(x.shape[0], C, h * p, h * p)

    def _x0_act(self, z):
        if self.x0_softmax:
            return 2.0 * z.softmax(dim=1) - 1.0
        return torch.tanh(z)


def CSGDiT_S_16(**kwargs):
    return CSGDiT_Segmentation(depth=4, hidden_size=512, num_heads=8, patch_size=16, **kwargs)

def CSGDiT_S_32(**kwargs):
    return CSGDiT_Segmentation(depth=4, hidden_size=512, num_heads=8, patch_size=32, **kwargs)

def CSGDiT_B_16(**kwargs):
    return CSGDiT_Segmentation(depth=6, hidden_size=768, num_heads=12, patch_size=16, **kwargs)

def CSGDiT_B_32(**kwargs):
    return CSGDiT_Segmentation(depth=6, hidden_size=768, num_heads=12, patch_size=32, **kwargs)

def CSGDiT_L_16(**kwargs):
    return CSGDiT_Segmentation(depth=8, hidden_size=1024, num_heads=16, patch_size=16, **kwargs)

def CSGDiT_L_32(**kwargs):
    return CSGDiT_Segmentation(depth=8, hidden_size=1024, num_heads=16, patch_size=32, **kwargs)

def CSGDiT_H_16(**kwargs):
    return CSGDiT_Segmentation(depth=12, hidden_size=1280, num_heads=16, patch_size=16, **kwargs)

def CSGDiT_H_32(**kwargs):
    return CSGDiT_Segmentation(depth=12, hidden_size=1280, num_heads=16, patch_size=32, **kwargs)


CSGDiT_models = {
    'CSGDiT-S/16': CSGDiT_S_16,
    'CSGDiT-S/32': CSGDiT_S_32,
    'CSGDiT-B/16': CSGDiT_B_16,
    'CSGDiT-B/32': CSGDiT_B_32,
    'CSGDiT-L/16': CSGDiT_L_16,
    'CSGDiT-L/32': CSGDiT_L_32,
    'CSGDiT-H/16': CSGDiT_H_16,
    'CSGDiT-H/32': CSGDiT_H_32,
}
