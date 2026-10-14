"""
The flow-matching transformer policy of the robosuite experiments (transfer/common.py build_policy and
diffusion_policy's FlowTransformerLowdimPolicy), for the 40-D KTaskPlace observation.

Training: x_t = t * noise + (1 - t) * action, the network predicts u = noise - action, t ~ Beta(1.5, 1)
on [0.001, 1]; sampling integrates from t = 1 to 0 in 10 Euler steps. Actions are normalized per dimension
(position to [-1, 1], 6D rotation and gripper as they are), observations per dimension to [-1, 1].
The policy acts on the last 2 observations and executes 8 of the 16 predicted actions.
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ktask.data import HORIZON, N_ACT, N_OBS
from ktask.env import ACT_DIM, OBS_DIM
from ktask.model import TransformerForDiffusion

N_STEPS = 10
A0, A1 = N_OBS - 1, N_OBS - 1 + N_ACT          # executed slice of the predicted actions


class Normalizer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer('scale', torch.ones(dim))
        self.register_buffer('offset', torch.zeros(dim))

    def fit(self, x, dims=None, eps=1e-4):
        """Map [min, max] of x to [-1, 1] on `dims` (default all); other dims stay as they are."""
        x = x.reshape(-1, x.shape[-1])
        lo, hi = x.min(0).values, x.max(0).values
        rng = hi - lo
        scale = torch.where(rng > eps, 2.0 / rng.clamp(min=eps), torch.ones_like(rng))
        offset = torch.where(rng > eps, -1.0 - scale * lo, -(hi + lo) / 2)
        dims = torch.arange(x.shape[-1]) if dims is None else torch.as_tensor(dims)
        self.scale[dims], self.offset[dims] = scale[dims].to(self.scale), offset[dims].to(self.offset)

    def normalize(self, x):
        return x * self.scale + self.offset

    def unnormalize(self, y):
        return (y - self.offset) / self.scale


class FlowPolicy(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM):
        super().__init__()
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.model = TransformerForDiffusion(
            input_dim=act_dim, output_dim=act_dim, horizon=HORIZON, n_obs_steps=N_OBS, cond_dim=obs_dim,
            n_layer=8, n_head=4, n_emb=256, p_drop_emb=0.0, p_drop_attn=0.3, causal_attn=True,
            time_as_cond=True, obs_as_cond=True, n_cond_layers=0)
        self.obs_norm = Normalizer(obs_dim)
        self.act_norm = Normalizer(act_dim)

    def fit_normalizers(self, obs, action):
        self.obs_norm.fit(obs)
        self.act_norm.fit(action, dims=[0, 1, 2])

    def cond(self, obs):
        """obs (B, >= N_OBS, obs_dim) raw -> normalized conditioning (B, N_OBS, obs_dim)."""
        return self.obs_norm.normalize(obs[:, :N_OBS])

    def loss(self, obs, action, weight=None):
        cond = self.cond(obs)
        traj = self.act_norm.normalize(action)
        noise = torch.randn_like(traj)
        t = torch.distributions.Beta(1.5, 1.0).sample((traj.shape[0],)).to(traj.device) * 0.999 + 0.001
        x_t = t[:, None, None] * noise + (1 - t[:, None, None]) * traj
        v = self.model(x_t, t * 999, cond)
        per = F.mse_loss(v, noise - traj, reduction='none').mean(dim=(1, 2))
        if weight is None:
            return per.mean()
        return (per * weight).sum() / weight.sum()

    @torch.no_grad()
    def sample(self, obs, gen=None):
        """Deterministic Euler integration from Gaussian noise; returns the executed actions (B, N_ACT, act_dim)."""
        cond = self.cond(obs)
        B = cond.shape[0]
        x = torch.randn(B, HORIZON, self.act_dim, device=cond.device, generator=gen)
        dt = -1.0 / N_STEPS
        for k in range(N_STEPS):
            t = torch.full((B,), (1.0 + k * dt) * 999, device=cond.device)
            x = x + dt * self.model(x, t, cond)
        return self.act_norm.unnormalize(x)[:, A0:A1]


class EMA:
    """diffusion_policy's EMAModel schedule (power 0.75, max 0.9999)."""

    def __init__(self, model, power=0.75, max_value=0.9999):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.power, self.max_value, self.step_count = power, max_value, 0

    @torch.no_grad()
    def step(self, model):
        step = max(0, self.step_count - 1)
        decay = 0.0 if step <= 0 else min(1 - (1 + step) ** -self.power, self.max_value)
        for p_ema, p in zip(self.model.parameters(), model.parameters()):
            p_ema.mul_(decay).add_(p.detach(), alpha=1 - decay)
        for b_ema, b in zip(self.model.buffers(), model.buffers()):
            b_ema.copy_(b)
        self.step_count += 1


def optimizer_for(policy, lr, weight_decay=1e-3):
    return torch.optim.AdamW(policy.model.get_optim_groups(weight_decay=weight_decay), lr=lr, betas=(0.9, 0.95))


def cosine_lr(step, total, warmup):
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


def save_checkpoint(path, policy, info):
    """The deployed (EMA) weights only: every later stage starts from and evaluates these."""
    import os
    tmp = path + '.tmp'
    torch.save({'policy': policy.state_dict(), 'info': info}, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, device='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    policy = FlowPolicy()
    policy.load_state_dict(payload['policy'])
    return policy.to(device), payload
