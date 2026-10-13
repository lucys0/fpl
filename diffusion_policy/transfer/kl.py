"""
KL between two flow policies through the noisy denoising chain PPO uses (transfer.ppo.Actor):
Euler integration with Gaussian noise of std `sigma` after every step, likelihood over the
executed slice of the action chunk. With the same sigma for both policies, each transition is
a Gaussian around x_k + dt * v(x_k), so on chains sampled from the reference policy

    KL(ref || pol) = sum_k E || dt * (v_pol(x_k) - v_ref(x_k)) ||^2 / (2 sigma^2)   (executed slice)

is exact up to sampling, and differentiable in the policy's parameters. Both policies share the
observation normalizer (fine-tuning keeps the base's).
"""

import torch

from transfer.common import ACTION_DIM, HORIZON, N_ACTION_STEPS, N_OBS_STEPS

A0, A1 = N_OBS_STEPS - 1, N_OBS_STEPS - 1 + N_ACTION_STEPS      # executed slice, as in transfer.ppo
SIGMA = 0.02                                                     # PPO_SPEC['sigma']


def cond_of(policy, obs):
    """Normalized conditioning from unnormalized observation windows (B, >=N_OBS_STEPS, obs_dim)."""
    return policy.normalizer['obs'].normalize(obs[:, :N_OBS_STEPS])


@torch.no_grad()
def ref_chain(ref, cond, sigma=SIGMA):
    """The reference policy's noisy chain: states x_k, timesteps and transition means, each (B, K, ...)."""
    model, K = ref.model, ref.num_inference_steps
    dt = -1.0 / K
    B = cond.shape[0]
    x = torch.randn(B, HORIZON, ACTION_DIM, device=cond.device)
    xs, ts, mus = [], [], []
    for k in range(K):
        t = torch.full((B,), (1.0 + k * dt) * 999, device=cond.device)
        mu = x + dt * model(x, t, cond)
        xs.append(x); ts.append(t); mus.append(mu)
        x = mu + sigma * torch.randn_like(mu)
    return torch.stack(xs, 1), torch.stack(ts, 1), torch.stack(mus, 1)


def chain_kl(pol, cond, chain, sigma=SIGMA):
    """KL(ref || pol) per decision, (B,), on chains from ref_chain; differentiable in pol."""
    xs, ts, mus_ref = chain
    B, K = ts.shape
    dt = -1.0 / K
    x = xs.reshape(B * K, HORIZON, ACTION_DIM)
    mu = x + dt * pol.model(x, ts.reshape(B * K), cond.repeat_interleave(K, 0))
    d = (mu.view(B, K, HORIZON, ACTION_DIM) - mus_ref)[:, :, A0:A1]
    return d.pow(2).sum(dim=(1, 2, 3)) / (2 * sigma ** 2)
