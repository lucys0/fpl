"""
Record a checkpoint's own episodes (the ktask.data format) on a set of tasks: deterministic flow sampling,
or with --sigma, PPO's stochastic chain (noise after every Euler step), as PPO's logged rollouts were.
Episodes end at their first success (that step included) or at MAX_STEPS.

  python -m ktask.collect CKPT -o out.npz --tasks 13 --n 4096 [--sigma 0.02] [--slots 1024]
"""

import time

import click
import numpy as np
import torch

import ktask.env  # noqa: F401
from ktask.data import N_ACT, parse_tasks, save_episodes
from ktask.env import task_name, to_env_action
from ktask.evaluate import MAX_STEPS, make_env
from ktask.policy import load_checkpoint


@torch.no_grad()
def record(env, act_fn):
    u = env.unwrapped
    n = u.num_envs
    o = u.policy_obs()
    hist = torch.stack([o, o], 1)
    alive = torch.ones(n, dtype=torch.bool, device=u.device)
    first = torch.full((n,), -1, dtype=torch.long, device=u.device)
    obs, act = [], []
    t = 0
    while t < MAX_STEPS and alive.any():
        actions = act_fn(hist)
        for j in range(N_ACT):
            cur = u.policy_obs()
            obs.append(cur)
            act.append(actions[:, j])
            _, _, _, _, info = env.step(to_env_action(actions[:, j]))
            new = info['success'] & alive
            first[new] = t
            alive &= ~new
            t += 1
            if t >= MAX_STEPS or not alive.any():
                break
        o = u.policy_obs()
        hist = torch.stack([cur, o], 1)                         # consecutive env steps, as in training
    obs, act = torch.stack(obs, 1).cpu().numpy(), torch.stack(act, 1).cpu().numpy()
    first = first.cpu().numpy()
    return obs, act, first


@click.command()
@click.argument('ckpt')
@click.option('-o', '--out', required=True)
@click.option('--tasks', required=True)
@click.option('--n', default=4096, help='episodes per task')
@click.option('--sigma', default=0.0)
@click.option('--slots', default=1024, help='env slots in all (split over the tasks)')
@click.option('--seed', default=7_000_000)
def main(ckpt, out, tasks, n, sigma, slots, seed):
    tasks = parse_tasks(tasks)
    per = slots // len(tasks)
    ids = [t for t in tasks for _ in range(per)]
    env = make_env(ids)
    u = env.unwrapped
    policy, _ = load_checkpoint(ckpt, u.device)
    policy.eval()
    gen = torch.Generator(device=u.device)
    if sigma > 0:
        from ktask.ppo import Actor
        actor = Actor(policy, sigma, [0, 10])
        act_fn = lambda h: actor.sample(policy.cond(h), gen)[0]
    else:
        act_fn = lambda h: policy.sample(h, gen)
    eps, t0 = [], time.time()
    for b in range((n + per - 1) // per):
        seeds = [seed + 100_000 * b + i for i in range(len(ids))]
        env.reset(seed=seeds)
        gen.manual_seed(seed + b)
        obs, act, first = record(env, act_fn)
        for i in range(len(ids)):
            if b * per + i % per >= n:
                continue
            ok = first[i] >= 0
            T = first[i] + 1 if ok else obs.shape[1]
            eps.append(dict(obs=obs[i, :T], action=act[i, :T], task=ids[i], q=-1.0, success=bool(ok),
                            first_success_step=int(first[i]), seed=seeds[i]))
        print(f'batch {b}: success {np.mean(first >= 0):.3f} ({time.time() - t0:.0f} s)', flush=True)
    save_episodes(out, eps, meta={'ckpt': ckpt, 'sigma': sigma})
    s = np.array([e['success'] for e in eps])
    tk = np.array([e['task'] for e in eps])
    for t in tasks:
        print(f'{task_name(t)}: success {s[tk == t].mean():.3f} over {np.sum(tk == t)} episodes')
    print(f'wrote {len(eps)} episodes to {out}')


if __name__ == '__main__':
    main()
