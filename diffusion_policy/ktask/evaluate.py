"""
Evaluate a checkpoint on a set of tasks (GPU simulation, deterministic flow sampling from seeded noise).

Episode i of task t starts from env seed EVAL_SEED + 1000 t + i, so every checkpoint meets the same start
states (paired comparisons). The policy sees the last two env-step observations, as in training (until
2026-10-14 evening the rollouts here, in ppo.py and collect.py, passed observations 8 steps apart). An episode ends at its first success or after MAX_STEPS env steps.
Output JSON: {"ckpt", "n", "tasks": {"<t>": {"success": [...], "steps": [...], "lifted": [...]}}}
(steps: env steps to success, -1 on failure).

  python -m ktask.evaluate CKPT -o eval.json [--tasks all] [--n 128] [--slots 32]
"""

import json
import os
import time

import click
import gymnasium as gym
import numpy as np
import torch

import ktask.env  # noqa: F401
from ktask.data import N_ACT, parse_tasks
from ktask.env import task_name, to_env_action
from ktask.policy import load_checkpoint

EVAL_SEED = 300_000
MAX_STEPS = 200


def make_env(task_ids):
    return gym.make('KTaskPlace-v0', num_envs=len(task_ids), task_ids=task_ids, obs_mode='state',
                    control_mode='pd_ee_pose', sim_backend='gpu', render_backend='none')


@torch.no_grad()
def run_batch(env, act_fn, max_steps=MAX_STEPS, history='step'):
    """Roll out act_fn(obs_hist (B, 2, 40)) -> actions (B, N_ACT, 10) from the current reset.
    history 'step': the two observations are consecutive env steps, as in training (and robomimic's
    MultiStepWrapper); 'chunk': the previous decision's observation and the current one, 8 steps apart
    (what every E1-E3 rollout used, by mistake; kept only to reproduce those numbers)."""
    u = env.unwrapped
    n = u.num_envs
    o = u.policy_obs()
    hist = torch.stack([o, o], 1)
    alive = torch.ones(n, dtype=torch.bool, device=u.device)
    steps = torch.full((n,), -1, dtype=torch.long, device=u.device)
    lifted = torch.zeros(n, dtype=torch.bool, device=u.device)
    t = 0
    while t < max_steps and alive.any():
        actions = act_fn(hist)
        prev = o
        for j in range(N_ACT):
            if history == 'step' and j == N_ACT - 1:
                prev = u.policy_obs()
            _, _, _, _, info = env.step(to_env_action(actions[:, j]))
            t += 1
            new = info['success'] & alive
            steps[new] = t
            lifted |= info['lifted'] & alive
            alive &= ~new
            if t >= max_steps or not alive.any():
                break
        o = u.policy_obs()
        hist = torch.stack([hist[:, 1] if history == 'chunk' else prev, o], 1)
    return steps >= 0, steps, lifted


def evaluate(policy, tasks, n, slots, seed=EVAL_SEED, log=print, sigma=0.0, history='step'):
    ids = [t for t in tasks for _ in range(slots)]
    env = make_env(ids)
    u = env.unwrapped
    gen = torch.Generator(device=u.device)
    res = {t: {'success': [], 'steps': [], 'lifted': []} for t in tasks}
    policy.eval()
    t0 = time.time()
    for b in range((n + slots - 1) // slots):
        env.reset(seed=[seed + 1000 * t + b * slots + i % slots for i, t in enumerate(ids)])
        gen.manual_seed(seed + b)
        if sigma > 0:                     # PPO's stochastic chain (noise sigma after every Euler step)
            from ktask.ppo import Actor
            actor = Actor(policy, sigma, [0, 10])
            act = lambda h: actor.sample(policy.cond(h), gen)[0]
        else:
            act = lambda h: policy.sample(h, gen)
        succ, steps, lifted = run_batch(env, act, history=history)
        succ, steps, lifted = succ.cpu().numpy(), steps.cpu().numpy(), lifted.cpu().numpy()
        for i, t in enumerate(ids):
            if b * slots + i % slots < n:
                res[t]['success'].append(bool(succ[i]))
                res[t]['steps'].append(int(steps[i]))
                res[t]['lifted'].append(bool(lifted[i]))
        log(f'batch {b}: success {succ.mean():.3f} ({time.time() - t0:.0f} s)')
    env.close()
    return res


@click.command()
@click.argument('ckpt')
@click.option('-o', '--out', required=True)
@click.option('--tasks', default='all')
@click.option('--n', default=128, help='episodes per task')
@click.option('--slots', default=32, help='env slots per task')
@click.option('--seed', default=EVAL_SEED)
@click.option('--sigma', default=0.0, help="evaluate PPO's stochastic policy (chain noise) instead")
@click.option('--history', default='step', type=click.Choice(['chunk', 'step']))
def main(ckpt, out, tasks, n, slots, seed, sigma, history):
    tasks = parse_tasks(tasks)
    policy, _ = load_checkpoint(ckpt, 'cuda')
    res = evaluate(policy, tasks, n, min(slots, n), seed, sigma=sigma, history=history)
    for t in tasks:
        r = res[t]
        print(f"{t:2d} {task_name(t):18s} success {np.mean(r['success']):.2f} lifted {np.mean(r['lifted']):.2f}")
    print(f"mean success {np.mean([np.mean(res[t]['success']) for t in tasks]):.3f}")
    tmp = out + '.tmp'
    json.dump({'ckpt': ckpt, 'n': n, 'seed': seed, 'tasks': {str(t): res[t] for t in tasks}}, open(tmp, 'w'))
    os.replace(tmp, out)


if __name__ == '__main__':
    main()
