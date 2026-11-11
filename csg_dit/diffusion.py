import torch
import torch.nn.functional as F
import numpy as np


class DiffusionSchedule:
    def __init__(self, num_timesteps=200, schedule_type='cosine'):
        self.num_timesteps = num_timesteps

        if schedule_type == 'cosine':
            self.betas = self._cosine_beta_schedule(num_timesteps)
        elif schedule_type == 'linear':
            self.betas = torch.linspace(1e-4, 0.02, num_timesteps)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    @staticmethod
    def _cosine_beta_schedule(timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    @staticmethod
    def _extract(a, t, x_shape):
        out = a.to(t.device).gather(0, t)
        return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_ac    = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_1m_ac = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_ac * x_start + sqrt_1m_ac * noise, noise

    @torch.no_grad()
    def ddim_sample(self, model, shape, image_cond, device,
                    ddim_steps=20, eta=0.0, progress=False, generator=None):
        if not 1 <= ddim_steps <= self.num_timesteps:
            raise ValueError('ddim_steps must lie between 1 and num_timesteps')
        batch_size = shape[0]

        condition = (model.encode_condition(image_cond)
                     if hasattr(model, 'encode_condition') else None)

        ddim_timesteps = np.rint(np.linspace(0, self.num_timesteps - 1,
                                             ddim_steps)).astype(np.int64)
        if ddim_steps == 1:
            ddim_timesteps[0] = self.num_timesteps - 1

        x_t = torch.randn(shape, device=device, generator=generator)

        steps = reversed(ddim_timesteps)
        if progress:
            from tqdm import tqdm
            steps = tqdm(steps, desc='DDIM Sampling', total=len(ddim_timesteps))

        for i, t in enumerate(steps):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            if condition is None:
                prediction = model(x_t, t_batch, image_cond)
            else:
                prediction = model(x_t, t_batch, image_cond, condition=condition)
            pred_x0 = prediction.clamp(-1.0, 1.0)

            alpha_t = self._extract(self.alphas_cumprod, t_batch, x_t.shape)
            if i < len(ddim_timesteps) - 1:
                t_prev = torch.full((batch_size,),
                                    ddim_timesteps[len(ddim_timesteps) - i - 2],
                                    device=device, dtype=torch.long)
                alpha_t_prev = self._extract(self.alphas_cumprod, t_prev, x_t.shape)
            else:
                alpha_t_prev = torch.ones_like(alpha_t)

            pred_noise = (x_t - torch.sqrt(alpha_t) * pred_x0) / torch.sqrt(1.0 - alpha_t)

            sigma = eta * torch.sqrt(
                (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)
            )
            noise = (torch.randn_like(x_t) if i < len(ddim_timesteps) - 1
                     else torch.zeros_like(x_t))
            dir_xt = torch.sqrt((1.0 - alpha_t_prev - sigma ** 2).clamp_min(0.0)) * pred_noise
            x_t = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt + sigma * noise

        return x_t


def dice_loss(pred, target, smooth=1e-5):
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return (1.0 - dice.mean(dim=1)).mean()


def compute_loss(model, diffusion, batch, device, use_dice=True, dice_weight=1.0,
                 ce_weight=0.0):
    image = batch['image'].to(device)
    mask  = batch['mask'].to(device)

    t = torch.randint(0, diffusion.num_timesteps, (mask.shape[0],), device=device).long()
    x_t, _ = diffusion.q_sample(mask, t)

    pred_x0 = model(x_t, t, image)

    loss = F.mse_loss(pred_x0, mask)

    if use_dice or ce_weight > 0.0:
        pred_01 = (pred_x0 + 1.0) / 2.0

        if use_dice:
            loss = loss + dice_weight * dice_loss(pred_01, (mask > 0.0).float())

        if ce_weight > 0.0:
            if mask.shape[1] == 1:
                raise ValueError('ce_weight > 0 needs a multi-class mask '
                                 '(mask_channels > 1) and the softmax x0 head')
            label = batch.get('label_idx')
            label = (mask.argmax(dim=1) if label is None else label.to(device)).long()
            loss = loss + ce_weight * F.nll_loss(torch.log(pred_01.clamp_min(1e-7)),
                                                 label)

    return loss, pred_x0, mask
