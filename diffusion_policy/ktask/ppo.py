"""
PPO fine-tuning of the flow policy on K tasks at once (DPPO-style, as transfer/ppo.py on robosuite).

Sampling with Gaussian noise (std sigma) after each Euler step in noise_steps turns action sampling into a
chain of Gaussian transitions with tractable log-probabilities; every chain step of a decision shares the
decision's advantage, log-probabilities cover the executed 8 actions. With chain_subsample m, each update
uses m random chain steps per decision (an unbiased estimate of the all-steps objective, m/10 the compute).
The critic gets its own critic_epochs over every batch, and the first critic_warmup iterations update only
the critic: a randomly initialized critic biases the early GAE advantages, which with 1,024-episode
iterations first collapsed the policy (pilot, 2026-10-14). Reward 1 on success, which ends the
episode; optional time_cost c subtracts c * (steps / MAX_STEPS) over an episode. Dropout stays off.

Each iteration runs one episode in each of n_envs slots, split evenly over the K tasks, so the K tasks get
equal shares of the episode budget. Advantages are normalized over the whole batch.

  python -m ktask.ppo spec.json
  {"out_dir", "init_ckpt", "tasks": [ids], "episodes": 65536, "save_at": [...], "n_envs": 1024,
   "sigma": 0.02, "noise_steps": [0, 10], "lr": 3e-6, "critic_lr": 1e-3, "epochs": 5, "minibatch": 512,
   "micro": 128, "chain_subsample": null, "critic_epochs": 5, "critic_warmup": 3, "clip": 0.1, "gamma": 0.99, "lam": 0.95, "target_kl": 0.05, "seed": 0, "time_cost": 0.0,
   "log_rollouts": false, "eval": {"tasks": "all", "n": 128}}
Saves ppo_ep<N>.ckpt at each save_at episode count (the ktask.train format) and, with "eval", evaluates each
one in a separate process (eval_ep<N>.json). With log_rollouts, every training episode is written to
rollouts_<M>_<N>.npz (episodes M+1..N, the ktask.data format) so offline methods can train on PPO's data.
"""

import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from ktask.data import HORIZON, N_ACT, save_episodes
from ktask.env import ACT_DIM, task_name, to_env_action
from ktask.evaluate import MAX_STEPS, make_env
from ktask.policy import A0, A1, N_STEPS, load_checkpoint, save_checkpoint


class Actor:
    """Noisy Euler sampling of the flow model and the log-probabilities of its chain."""

    def __init__(self, policy, sigma, noise_steps):
        self.policy, self.model = policy, policy.model
        self.dt = -1.0 / N_STEPS
        self.sigma = sigma
        self.noisy = list(range(*noise_steps))

    @torch.no_grad()
    def sample(self, cond, gen=None):
        B = cond.shape[0]
        x = torch.randn(B, HORIZON, ACT_DIM, device=cond.device, generator=gen)
        xs = [x]
        for k in range(N_STEPS):
            t = torch.full((B,), (1.0 + k * self.dt) * 999, device=cond.device)
            x = x + self.dt * self.model(x, t, cond)
            if k in self.noisy:
                x = x + self.sigma * torch.randn(x.shape, device=x.device, generator=gen)
            xs.append(x)
        xs = torch.stack(xs, 1)                                     # (B, K+1, T, Da)
        return self.policy.act_norm.unnormalize(xs[:, -1])[:, A0:A1], xs

    def logprob(self, cond, xs, pos=None):
        """log pi(x_{k+1} | x_k) over the noisy chain steps: (B, len(noisy)); with pos (B, m), only at the
        noisy steps self.noisy[pos] of each sample: (B, m)."""
        B = xs.shape[0]
        noisy = torch.tensor(self.noisy, device=xs.device)
        ks = noisy[None].expand(B, -1) if pos is None else noisy[pos]
        n = ks.shape[1]
        rows = torch.arange(B, device=xs.device)[:, None].expand(-1, n)
        x_prev = xs[rows, ks].reshape(B * n, HORIZON, ACT_DIM)
        x_next = xs[rows, ks + 1].reshape(B * n, HORIZON, ACT_DIM)
        t = ((1.0 + ks * self.dt) * 999).float().reshape(-1)
        mean = x_prev + self.dt * self.model(x_prev, t, cond.repeat_interleave(n, 0))
        z = (x_next - mean)[:, A0:A1] / self.sigma
        lp = -0.5 * z.pow(2) - math.log(self.sigma) - 0.5 * math.log(2 * math.pi)
        return lp.sum(dim=(1, 2)).view(B, n)


def batched(fn, *xs, size=4096):
    return torch.cat([fn(*(x[i:i + size] for x in xs)) for i in range(0, len(xs[0]), size)])


def main(spec_path):
    spec = json.load(open(spec_path))
    out_dir = spec['out_dir']
    os.makedirs(out_dir, exist_ok=True)
    log_f = open(os.path.join(out_dir, 'ppo_log.txt'), 'a')

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_f.write(line + '\n')
        log_f.flush()

    seed = spec.get('seed', 0)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False      # chain log-probabilities need full fp32 precision
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    policy, _ = load_checkpoint(spec['init_ckpt'], device)
    policy.eval()                                                  # dropout off throughout
    actor = Actor(policy, spec.get('sigma', 0.02), spec.get('noise_steps', [0, N_STEPS]))
    critic = nn.Sequential(nn.Linear(2 * policy.obs_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                           nn.Linear(256, 1)).to(device)
    opt_actor = torch.optim.Adam(policy.model.parameters(), lr=spec.get('lr', 3e-6))
    opt_critic = torch.optim.Adam(critic.parameters(), lr=spec.get('critic_lr', 1e-3))

    tasks = [int(t) for t in spec['tasks']]
    n_envs = spec.get('n_envs', 1024)
    assert n_envs % len(tasks) == 0, (n_envs, len(tasks))
    ids = [t for t in tasks for _ in range(n_envs // len(tasks))]
    env = make_env(ids)
    u = env.unwrapped
    task_of = torch.tensor(ids, device=device)
    step_cost = spec.get('time_cost', 0.0) * N_ACT / MAX_STEPS
    gamma, lam, clip = spec.get('gamma', 0.99), spec.get('lam', 0.95), spec.get('clip', 0.1)
    save_at = sorted(spec.get('save_at', [spec['episodes']]))
    log_rollouts, logged, logged_from = spec.get('log_rollouts', False), [], 0
    evals = []
    episodes_done, it, t0 = 0, 0, time.time()
    log(f"PPO on {len(tasks)} tasks ({', '.join(task_name(t) for t in tasks)}), {n_envs} slots, "
        f"{spec['episodes']} episodes")

    while episodes_done < spec['episodes']:
        # ---------------- collect one episode per slot
        t_iter = time.time()
        seeds = [5_000_000 + 100_000 * seed + episodes_done + i for i in range(n_envs)]
        env.reset(seed=seeds)
        o = u.policy_obs()
        hist = torch.stack([o, o], 1)
        alive = torch.ones(n_envs, dtype=torch.bool, device=device)
        success = torch.zeros(n_envs, dtype=torch.bool, device=device)
        first = torch.full((n_envs,), -1, dtype=torch.long, device=device)
        buf = {k: [] for k in ('cond', 'xs', 'reward', 'alive')}
        traj_obs, traj_act = [], []
        t_env = 0
        for _ in range(math.ceil(MAX_STEPS / N_ACT)):
            cond = policy.cond(hist)
            action, xs = actor.sample(cond, gen)
            reward = torch.zeros(n_envs, device=device)
            was_alive = alive.clone()
            for j in range(N_ACT):
                if log_rollouts:
                    traj_obs.append(u.policy_obs())
                    traj_act.append(action[:, j])
                _, _, _, _, info = env.step(to_env_action(action[:, j]))
                t_env += 1
                new = info['success'] & alive
                reward += new.float()
                first[new] = t_env - 1
                success |= new
                alive &= ~new
            reward -= step_cost * was_alive.float()
            for k, v in zip(buf, (cond, xs, reward, was_alive)):
                buf[k].append(v)
            if not alive.any():
                break
            o = u.policy_obs()
            hist = torch.stack([hist[:, 1], o], 1)
        if log_rollouts:
            obs_all = torch.stack(traj_obs, 1).cpu().numpy()
            act_all = torch.stack(traj_act, 1).cpu().numpy()
            fs, sc = first.cpu().numpy(), success.cpu().numpy()
            for i in range(n_envs):
                T = fs[i] + 1 if sc[i] else min(t_env, MAX_STEPS)
                logged.append(dict(obs=obs_all[i, :T], action=act_all[i, :T], task=ids[i], q=-1.0,
                                   success=bool(sc[i]), first_success_step=int(fs[i]), seed=seeds[i],
                                   ppo_iter=it + 1))
        episodes_done += n_envs
        it += 1

        # ---------------- advantages (GAE over decisions; episodes end at success or the time limit)
        t_roll = time.time() - t_iter
        T = len(buf['reward'])
        cond_t = torch.stack(buf['cond'])                                  # (T, n, 2, obs)
        with torch.no_grad():
            val = critic(cond_t.flatten(2)).squeeze(-1)                   # (T, n)
        rew, mask = torch.stack(buf['reward']), torch.stack(buf['alive']).float()
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
        cond, xs = flat(cond_t), flat(torch.stack(buf['xs']))
        adv, ret = flat(adv), flat(ret)
        task_s = flat(task_of[None].expand(T, -1))
        with torch.no_grad():
            logp_old = batched(actor.logprob, cond, xs, size=1024)
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)

        # ---------------- PPO update
        n = len(adv)
        mb, micro = spec.get('minibatch', 512), spec.get('micro', 128)
        n_noisy = len(actor.noisy)
        n_sub = min(spec.get('chain_subsample') or n_noisy, n_noisy)
        micro = max(1, micro * n_noisy // n_sub)              # micro counts decisions x chain steps
        kls, clipfrac, stop, n_updates = [], [], False, 0
        # critic: its own epochs over the whole batch (not cut short by the actor's KL early stop)
        cobs = cond.flatten(1)
        for epoch in range(spec.get('critic_epochs', 5)):
            perm = torch.randperm(n, device=device, generator=gen)
            for s in range(0, n, 512):
                b = perm[s:s + 512]
                v_loss = (critic(cobs[b]).squeeze(-1) - ret[b]).pow(2).mean()
                opt_critic.zero_grad()
                v_loss.backward()
                opt_critic.step()
        warm = it <= spec.get('critic_warmup', 0)         # first iterations: fit the critic only
        for epoch in range(0 if warm else spec.get('epochs', 5)):
            perm = torch.randperm(n, device=device, generator=gen)
            for s in range(0, n, mb):
                idx = perm[s:s + mb]
                opt_actor.zero_grad()
                lrs, rs = [], []
                for c in range(0, len(idx), micro):        # gradient accumulation over micro-batches
                    sub = idx[c:c + micro]
                    if n_sub < n_noisy:                     # a random subset of chain steps per sample
                        pos = torch.rand(len(sub), n_noisy, device=device, generator=gen).argsort(1)[:, :n_sub]
                        lp = actor.logprob(cond[sub], xs[sub], pos)
                        log_ratio = lp - logp_old[sub].gather(1, pos)
                    else:
                        lp = actor.logprob(cond[sub], xs[sub])
                        log_ratio = lp - logp_old[sub]
                    ratio = log_ratio.exp()
                    a = adv_n[sub][:, None]
                    loss = -torch.min(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a).sum() / (len(idx) * lp.shape[1])
                    loss.backward()
                    lrs.append(log_ratio.detach())
                    rs.append(ratio.detach())
                nn.utils.clip_grad_norm_(policy.model.parameters(), 1.0)
                opt_actor.step()
                log_ratio, ratio = torch.cat(lrs), torch.cat(rs)
                n_updates += 1
                with torch.no_grad():
                    kls.append(((ratio - 1) - log_ratio).mean().item())
                    clipfrac.append(((ratio - 1).abs() > clip).float().mean().item())
                if kls[-1] > spec.get('target_kl', 0.05):
                    stop = True
                    break
            if stop:
                break
        per_task = ' '.join(f'{success[task_of == t].float().mean().item():.2f}' for t in tasks) if len(tasks) <= 16 else ''
        log(f"iter {it} episodes {episodes_done}: success {success.float().mean().item():.3f} [{per_task}], "
            f"decisions {n}, updates {n_updates}, kl {np.mean(kls) if kls else 0:.4f}, "
            f"clipfrac {np.mean(clipfrac) if clipfrac else 0:.2f}, {'(critic warmup) ' if warm else ''}"
            f"value loss {v_loss.item():.4f}{' (early stop)' if stop else ''}, rollout {t_roll:.0f} s, "
            f"update {time.time() - t_iter - t_roll:.0f} s, {(time.time() - t0) / 60:.1f} min, "
            f"GPU mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")

        while save_at and episodes_done >= save_at[0]:
            n_ep = save_at.pop(0)
            path = os.path.join(out_dir, f'ppo_ep{n_ep}.ckpt')
            save_checkpoint(path, policy, {'spec': spec, 'episodes': episodes_done})
            log(f'saved {path}')
            if log_rollouts:
                part = os.path.join(out_dir, f'rollouts_{logged_from}_{n_ep}.npz')
                save_episodes(part, logged, meta={'spec': spec})
                log(f"wrote {len(logged)} episodes ({np.mean([e['success'] for e in logged]):.2f} successful) to {part}")
                logged, logged_from = [], n_ep
            if spec.get('eval'):
                ev = spec['eval']
                out = os.path.join(out_dir, f'eval_ep{n_ep}.json')
                cmd = [sys.executable, '-m', 'ktask.evaluate', path, '-o', out, '--tasks', str(ev.get('tasks', 'all')),
                       '--n', str(ev.get('n', 128))]
                evals.append(subprocess.Popen(cmd, stdout=open(out + '.log', 'w'), stderr=subprocess.STDOUT))
    env.close()
    for p in evals:
        p.wait()
    bad = [p.args for p in evals if p.returncode]
    if bad:
        log(f'{len(bad)} evaluations failed: {bad}')
        sys.exit(1)
    log('done')


if __name__ == '__main__':
    main(sys.argv[1])
