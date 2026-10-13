"""
PPO fine-tuning of the goal-conditioned flow policy on one task (DPPO-style).

Sampling the flow policy with Gaussian noise (std `sigma`) added after the
Euler integration steps in `noise_steps` ([start, end)) turns action sampling into a chain of
Gaussian transitions with tractable log-probabilities (as DPPO, Ren et al.
2024, does for diffusion policies). Every chain step of a decision shares that
decision's advantage; log-probabilities cover the executed part of the action
chunk. Reward is 1 when the nut lands on the commanded peg, which ends the
episode; landing on the other peg ends it with 0. Dropout stays off so rollout
and update log-probabilities agree.

Usage: python -m transfer.ppo spec.json
  {"out_dir": ..., "init_ckpt": ..., "peg": "left", "episodes": 1000, "save_at": [200, 1000],
   "n_envs": 30, "sigma": 0.1, "noise_steps": [0, 6], "lr": 1e-5, "critic_lr": 1e-3, "epochs": 5,
   "minibatch": 256, "clip": 0.1, "gamma": 0.99, "lam": 0.95, "target_kl": 0.05, "seed": 0,
   "env_meta_from": <hdf5 with env_args>, "nut_range": [[xlo, ylo], [xhi, yhi]] or null}
Checkpoints at each `save_at` episode count (ppo_ep<N>.ckpt) use the same format
as transfer.train, so transfer.rollout evaluates them (deterministically).
"""

import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv

from transfer.common import (ACTION_DIM, HORIZON, MAX_STEPS, N_ACTION_STEPS, N_OBS_STEPS, env_meta_from,
                             goal_vec, load_checkpoint, make_env_fn, save_checkpoint, to_env_action)

PEG_ID = {'left': 1, 'right': 2}
A0, A1 = N_OBS_STEPS - 1, N_OBS_STEPS - 1 + N_ACTION_STEPS      # executed slice of the action chunk


class Actor:
    """Noisy Euler sampling of the flow model and the log-probabilities of its chain."""

    def __init__(self, policy, sigma, noise_steps):
        """noise_steps: [start, end) of the integration steps followed by noise. Use all of
        them: the network is shared across steps, so updates also change any deterministic
        steps, outside the PPO ratio (noise on steps 0-5 only collapsed the policy)."""
        self.policy, self.model = policy, policy.model
        self.K = policy.num_inference_steps
        self.dt = -1.0 / self.K
        self.sigma = sigma
        self.noisy = list(range(*noise_steps))

    def cond(self, obs):
        return self.policy.normalizer['obs'].normalize(obs)[:, :N_OBS_STEPS]

    @torch.no_grad()
    def sample(self, cond):
        B = cond.shape[0]
        x = torch.randn(B, HORIZON, ACTION_DIM, device=cond.device)
        xs = [x]
        for k in range(self.K):
            t = torch.full((B,), (1.0 + k * self.dt) * 999, device=cond.device)
            x = x + self.dt * self.model(x, t, cond)
            if k in self.noisy:
                x = x + self.sigma * torch.randn_like(x)
            xs.append(x)
        xs = torch.stack(xs, 1)                                     # (B, K+1, T, Da)
        action = self.policy.normalizer['action'].unnormalize(xs[:, -1])[:, A0:A1]
        return action, xs

    def logprob(self, cond, xs):
        """log pi(x_{k+1} | x_k) over the noisy chain steps: (B, len(noisy))."""
        B, n = xs.shape[0], len(self.noisy)
        ks = torch.tensor(self.noisy, device=xs.device)
        x_prev = xs[:, ks].reshape(B * n, HORIZON, ACTION_DIM)
        x_next = xs[:, ks + 1].reshape(B * n, HORIZON, ACTION_DIM)
        t = ((1.0 + ks * self.dt) * 999).float().repeat(B)
        mean = x_prev + self.dt * self.model(x_prev, t, cond.repeat_interleave(n, 0))
        z = (x_next - mean)[:, A0:A1] / self.sigma
        lp = -0.5 * z.pow(2) - math.log(self.sigma) - 0.5 * math.log(2 * math.pi)
        return lp.sum(dim=(1, 2)).view(B, n)


def main(spec_path):
    spec = json.load(open(spec_path))
    out_dir = spec['out_dir']
    os.makedirs(out_dir, exist_ok=True)
    log_f = open(os.path.join(out_dir, 'ppo_log.txt'), 'a')

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True); log_f.write(line + '\n'); log_f.flush()

    seed = spec.get('seed', 0)
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device('cuda')
    _, policy, payload = load_checkpoint(spec['init_ckpt'], device)     # start from the EMA weights
    assert payload['extra_cond_dim'] == 0
    policy.eval()                                                      # dropout off throughout
    actor = Actor(policy, spec.get('sigma', 0.1), spec.get('noise_steps', [0, 6]))
    peg = spec['peg']
    goal = torch.from_numpy(goal_vec(peg)).to(device)

    obs_dim = N_OBS_STEPS * policy.obs_dim
    critic = nn.Sequential(nn.Linear(obs_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                           nn.Linear(256, 1)).to(device)
    opt_actor = torch.optim.Adam(policy.model.parameters(), lr=spec.get('lr', 1e-5))
    opt_critic = torch.optim.Adam(critic.parameters(), lr=spec.get('critic_lr', 1e-3))

    n_envs = spec.get('n_envs', 30)
    env = AsyncVectorEnv([make_env_fn(env_meta_from(spec['env_meta_from']), nut_range=spec.get('nut_range'))] * n_envs)
    gamma, lam, clip = spec.get('gamma', 0.99), spec.get('lam', 0.95), spec.get('clip', 0.1)
    save_at = sorted(spec.get('save_at', [spec['episodes']]))
    episodes_done, it, t0 = 0, 0, time.time()
    hist = []

    while episodes_done < spec['episodes']:
        # ---------------- collect one batch of episodes
        env.seed([5_000_000 + 100_000 * seed + episodes_done + i for i in range(n_envs)])
        obs = env.reset()
        alive = np.ones(n_envs, dtype=bool)
        buf = {k: [] for k in ('cond', 'xs', 'logp', 'value', 'reward', 'alive')}
        success = np.zeros(n_envs, dtype=bool)
        for _ in range(math.ceil(MAX_STEPS / N_ACTION_STEPS)):
            o = torch.from_numpy(obs.astype(np.float32)).to(device)
            o = torch.cat([o, goal.expand(o.shape[0], o.shape[1], -1)], dim=-1)
            with torch.no_grad():
                cond = actor.cond(o)
                action, xs = actor.sample(cond)
                logp = actor.logprob(cond, xs)
                value = critic(cond.flatten(1)).squeeze(-1)
            obs, _, done, _ = env.step(to_env_action(action.cpu().numpy()))
            first = env.call('get_first_peg')
            reward = np.zeros(n_envs, dtype=np.float32)
            ended = np.zeros(n_envs, dtype=bool)
            for i, (p, _) in enumerate(first):
                if alive[i] and p:
                    ended[i] = True
                    if p == PEG_ID[peg]:
                        reward[i], success[i] = 1.0, True
            for k, v in zip(buf, (cond, xs, logp, value, torch.from_numpy(reward).to(device),
                                  torch.from_numpy(alive.copy()).to(device))):
                buf[k].append(v)
            alive &= ~ended & ~np.asarray(done, dtype=bool)
            if not alive.any():
                break
        episodes_done += n_envs
        it += 1

        # ---------------- advantages (GAE over decisions; episodes end at success, wrong peg, or time limit)
        T = len(buf['reward'])
        rew, val = torch.stack(buf['reward']), torch.stack(buf['value'])      # (T, n_envs)
        mask = torch.stack(buf['alive']).float()
        adv = torch.zeros_like(rew)
        last = torch.zeros(n_envs, device=device)
        for t in reversed(range(T)):
            nonterminal = mask[t + 1] if t + 1 < T else torch.zeros(n_envs, device=device)
            next_val = val[t + 1] if t + 1 < T else torch.zeros(n_envs, device=device)
            delta = rew[t] + gamma * next_val * nonterminal - val[t]
            last = delta + gamma * lam * nonterminal * last
            adv[t] = last
        ret = adv + val
        valid = mask.bool().flatten()
        flat = lambda x: x.flatten(0, 1)[valid]
        cond, xs, logp_old = flat(torch.stack(buf['cond'])), flat(torch.stack(buf['xs'])), flat(torch.stack(buf['logp']))
        adv, ret = flat(adv), flat(ret)
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)

        # ---------------- PPO update
        n = len(adv)
        mb = spec.get('minibatch', 256)
        kls, clipfrac, stop = [], [], False
        for epoch in range(spec.get('epochs', 5)):
            perm = torch.randperm(n, device=device)
            for s in range(0, n, mb):
                idx = perm[s:s + mb]
                lp = actor.logprob(cond[idx], xs[idx])
                log_ratio = lp - logp_old[idx]
                ratio = log_ratio.exp()
                a = adv_n[idx][:, None]
                loss = -torch.min(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a).mean()
                opt_actor.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(policy.model.parameters(), 1.0)
                opt_actor.step()
                v_loss = (critic(cond[idx].flatten(1)).squeeze(-1) - ret[idx]).pow(2).mean()
                opt_critic.zero_grad(); v_loss.backward(); opt_critic.step()
                with torch.no_grad():
                    kls.append(((ratio - 1) - log_ratio).mean().item())
                    clipfrac.append(((ratio - 1).abs() > clip).float().mean().item())
                if kls[-1] > spec.get('target_kl', 0.05):     # stop this iteration's updates early
                    stop = True
                    break
            if stop:
                break
        hist.append(success.mean())
        log(f"iter {it} episodes {episodes_done}: train success {success.mean():.2f} "
            f"(last 5 iters {np.mean(hist[-5:]):.2f}), decisions {n}, kl {np.mean(kls):.4f}, "
            f"clipfrac {np.mean(clipfrac):.2f}, value loss {v_loss.item():.4f}"
            + (' (early stop)' if stop else '') + f", {(time.time() - t0) / 60:.1f} min")

        while save_at and episodes_done >= save_at[0]:
            path = os.path.join(out_dir, f'ppo_ep{save_at[0]}.ckpt')
            save_checkpoint(path, policy, policy, 0, {'spec': spec, 'episodes': episodes_done})
            log(f"saved {path}")
            save_at.pop(0)
    env.close()


if __name__ == '__main__':
    main(sys.argv[1])
