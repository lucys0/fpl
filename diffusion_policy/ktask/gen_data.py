"""
Record scripted-operator episodes for a set of tasks (every attempt is kept, as in transfer/gen_mixed.py).

Each episode draws its quality q ~ U(q_lo, q_hi) (q = 0: the clean expert). An episode ends at its first
success (that step included) or, failing that, 10 steps after the operator finishes its plan, or at
--max_steps. Output .npz (see ktask.data): per step obs (40) and action (10); per episode task, q, success,
first_success_step, length, seed.

  python -m ktask.gen_data -o data/mixed.npz --tasks all --n 400 --q_lo 0 --q_hi 1.5 --seed 0
"""

import time

import click
import gymnasium as gym
import numpy as np
import torch

import ktask.env  # noqa: F401
from ktask.data import parse_tasks, save_episodes
from ktask.env import N_TASKS, task_name, to_env_action
from ktask.expert import Expert

IDLE_AFTER_PLAN = 10


def record(env, q, gen, max_steps):
    """Run one batch of episodes from the current reset; returns per-slot lists of obs/actions and outcomes."""
    u = env.unwrapped
    ex = Expert(u, q, gen)
    n = u.num_envs
    obs, act = [], []
    alive = torch.ones(n, dtype=torch.bool, device=u.device)
    length = torch.zeros(n, dtype=torch.long, device=u.device)
    success = torch.zeros(n, dtype=torch.bool, device=u.device)
    first = torch.full((n,), -1, dtype=torch.long, device=u.device)
    done_for = torch.zeros(n, dtype=torch.long, device=u.device)
    for t in range(max_steps):
        o = u.policy_obs()
        a = ex.act()
        obs.append(o)
        act.append(a)
        length += alive.long()
        _, _, _, _, info = env.step(to_env_action(a))
        new = info['success'] & alive
        success |= new
        first[new] = t
        done_for = torch.where(ex.done, done_for + 1, done_for)
        alive &= ~new & (done_for < IDLE_AFTER_PLAN)
        if not alive.any():
            break
    obs, act = torch.stack(obs, 1).cpu().numpy(), torch.stack(act, 1).cpu().numpy()
    length = length.cpu().numpy()
    return [(obs[i, :length[i]], act[i, :length[i]]) for i in range(n)], success.cpu().numpy(), first.cpu().numpy()


@click.command()
@click.option('-o', '--out', required=True)
@click.option('--tasks', default='all', help="'all' or comma-separated task ids")
@click.option('--n', default=400, help='episodes per task')
@click.option('--q_lo', default=0.0)
@click.option('--q_hi', default=1.5)
@click.option('--slots', default=32, help='env slots per task per batch')
@click.option('--max_steps', default=200)
@click.option('--seed', default=0)
def main(out, tasks, n, q_lo, q_hi, slots, max_steps, seed):
    tasks = parse_tasks(tasks)
    ids = [t for t in tasks for _ in range(slots)]
    env = gym.make('KTaskPlace-v0', num_envs=len(ids), task_ids=ids, obs_mode='state', control_mode='pd_ee_pose',
                   sim_backend='gpu', render_backend='none')
    u = env.unwrapped
    gen = torch.Generator(device=u.device)
    rng = np.random.RandomState(seed)
    eps = []
    t0 = time.time()
    for b in range((n + slots - 1) // slots):
        seeds = [10_000_000 * (seed + 1) + 1000 * b * N_TASKS + i for i in range(len(ids))]
        env.reset(seed=seeds)
        gen.manual_seed(seed * 100_000 + b)
        q = torch.from_numpy(rng.uniform(q_lo, q_hi, size=len(ids))).float()
        trajs, succ, first = record(env, q, gen, max_steps)
        for i, (o, a) in enumerate(trajs):
            if b * slots + i % slots < n:
                eps.append(dict(obs=o, action=a, task=ids[i], q=float(q[i]), success=bool(succ[i]),
                                first_success_step=int(first[i]), seed=seeds[i]))
        print(f'batch {b}: success {succ.mean():.2f}, {time.time() - t0:.0f} s', flush=True)
    save_episodes(out, eps, meta=dict(q_lo=q_lo, q_hi=q_hi, seed=seed, max_steps=max_steps))
    s = np.array([e['success'] for e in eps])
    tk = np.array([e['task'] for e in eps])
    for t in tasks:
        print(f'{t:2d} {task_name(t):18s} success {s[tk == t].mean():.2f}')
    print(f'wrote {len(eps)} episodes ({s.mean():.2f} successful) to {out}')


if __name__ == '__main__':
    main()
