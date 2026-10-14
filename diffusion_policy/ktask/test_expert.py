"""Run the scripted operator on every task: success rate and episode length per task, and throughput.

  python -m ktask.test_expert [--n 4] [--q 0.0] [--steps 200]
"""
import time

import click
import gymnasium as gym
import numpy as np
import torch

import ktask.env  # noqa: F401  (registers KTaskPlace-v0)
from ktask.env import N_TASKS, task_name, to_env_action
from ktask.expert import Expert


@click.command()
@click.option('--n', default=4, help='env slots per task')
@click.option('--q', default=0.0)
@click.option('--steps', default=200)
@click.option('--seed', default=0)
def main(n, q, steps, seed):
    ids = [t for t in range(N_TASKS) for _ in range(n)]
    t0 = time.time()
    env = gym.make('KTaskPlace-v0', num_envs=len(ids), task_ids=ids, obs_mode='state', control_mode='pd_ee_pose',
                   sim_backend='gpu', render_backend='none')
    u = env.unwrapped
    print(f'built {len(ids)} envs in {time.time() - t0:.1f} s', flush=True)
    env.reset(seed=list(range(seed, seed + len(ids))))
    gen = torch.Generator(device=u.device)
    gen.manual_seed(seed)
    ex = Expert(u, torch.full((len(ids),), q), gen)
    succ = torch.zeros(len(ids), dtype=torch.bool, device=u.device)
    first = torch.full((len(ids),), -1, device=u.device)
    grasped = torch.zeros_like(succ)
    torch.cuda.synchronize()
    t0 = time.time()
    for t in range(steps):
        a = ex.act()
        _, _, _, _, info = env.step(to_env_action(a))
        new = info['success'] & ~succ
        first[new] = t + 1
        succ |= info['success']
        grasped |= info['lifted']
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f'{steps} steps x {len(ids)} envs in {dt:.1f} s: {steps * len(ids) / dt:.0f} env-steps/s', flush=True)
    s, f, g = succ.cpu().numpy(), first.cpu().numpy(), grasped.cpu().numpy()
    ph = ex.phase.cpu().numpy()
    for task in range(N_TASKS):
        m = np.arange(len(ids)) // n == task
        lens = f[m][s[m]]
        print(f'{task:2d} {task_name(task):18s} success {s[m].mean():.2f} lifted {g[m].mean():.2f} '
              f'steps {lens.mean() if len(lens) else float("nan"):5.1f}  phases {ph[m].tolist()}')
    print(f'overall success {s.mean():.3f}, lifted {g.mean():.3f}')


if __name__ == '__main__':
    main()
