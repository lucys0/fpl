"""
Composition eval for the Bimodal Square (slow_fast) task, after Fig. 4 and
Table 6 of the FPL paper: roll out a reward-conditioned policy under each
conditioning target and record which peg the nut lands on and the first step at
which it is placed. The demos only place the nut on the right peg slowly, so
FPL placing it on the right peg *fast* is the composed behavior.

Rollouts use env seeds starting at --test_start_seed (default 200000), disjoint
from the 100000+ seeds the training-time eval uses to pick the best checkpoint.

Usage:
  python scripts/eval_composition.py --ckpt <ckpt> --method fpl --output_dir <dir>
"""

import sys
import os
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import json
import click
import dill
import hydra
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf

from diffusion_policy.env_runner.reward_conditioned_lowdim_runner import RewardConditionedLowdimRunner

# Conditioning targets in normalized score units ([-1, 1]); the magnitudes match
# EVAL_Z_POSITIVE / EVAL_Z_NEGATIVE in scripts/slow_fast/launch_*.sh.
CONDITIONS = {
    # FPL conditions on [speed, peg]; peg +1 = right peg, -1 = left peg
    'fpl': {
        'fast_right': [0.9, 0.8],
        'slow_right': [-0.9, 0.8],
        'fast_left': [0.9, -0.8],
        'slow_left': [-0.9, -0.8],
    },
    # Single preference conditions on one composite score (mean of speed and peg)
    'single_pref': {
        'high': [0.8],
        'zero': [0.0],
        'low': [-0.8],
    },
}


def load_policy(ckpt, device):
    payload = torch.load(open(ckpt, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    workspace = hydra.utils.get_class(cfg._target_)(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(device)
    policy.eval()
    return policy, cfg


def rollout_summary(first_steps_left, first_steps_right, n_rollouts):
    """Paper-style metrics from per-rollout first-success steps (successes only).

    Success / T1S / throughput follow the paper's standard direction, where
    success means the nut ends on the right peg; throughput is 300 / T1S per
    rollout (0 for rollouts that don't succeed), averaged over all rollouts.
    """
    left = np.asarray(first_steps_left, dtype=float)
    right = np.asarray(first_steps_right, dtype=float)
    return {
        'right_peg_success': len(right) / n_rollouts,
        'left_peg_success': len(left) / n_rollouts,
        'any_peg_success': (len(left) + len(right)) / n_rollouts,
        't1s_right': float(right.mean()) if len(right) else None,
        't1s_left': float(left.mean()) if len(left) else None,
        'throughput_right': float((300.0 / right).sum() / n_rollouts),
        'throughput_left': float((300.0 / left).sum() / n_rollouts),
    }


@click.command()
@click.option('--ckpt', required=True, help='Policy checkpoint to evaluate')
@click.option('--method', required=True, type=click.Choice(sorted(CONDITIONS)))
@click.option('--output_dir', required=True)
@click.option('--n_rollouts', default=50, type=int, help='Rollouts per condition')
@click.option('--n_videos', default=2, type=int, help='Videos recorded per condition')
@click.option('--n_envs', default=None, type=int, help='Parallel envs (default: the training config)')
@click.option('--test_start_seed', default=200000, type=int)
@click.option('--wandb_project', default='slow_fast_composition_eval')
@click.option('--run_name', default=None)
@click.option('--device', default='cuda:0')
def main(ckpt, method, output_dir, n_rollouts, n_videos, n_envs, test_start_seed,
         wandb_project, run_name, device):
    os.makedirs(output_dir, exist_ok=True)
    conditions = CONDITIONS[method]
    policy, cfg = load_policy(ckpt, torch.device(device))
    num_reward_dims = int(cfg.num_reward_dims)
    for name, z in conditions.items():
        assert len(z) == num_reward_dims, f"{name}: {z} vs num_reward_dims={num_reward_dims}"

    runner_kwargs = OmegaConf.to_container(cfg.task.env_runner, resolve=True)
    for key in ('_target_', 'target_rewards', 'num_reward_dims'):
        runner_kwargs.pop(key, None)
    runner_kwargs.update(output_dir=output_dir, n_train=0, n_train_vis=0,
                         n_test=n_rollouts, n_test_vis=n_videos,
                         test_start_seed=test_start_seed)
    if n_envs is not None:
        runner_kwargs['n_envs'] = n_envs
    runner = RewardConditionedLowdimRunner(
        target_rewards=list(conditions.values())[0],
        num_reward_dims=num_reward_dims, **runner_kwargs)

    wandb.init(project=wandb_project, name=run_name, config={
        'ckpt': ckpt, 'method': method, 'conditions': conditions,
        'n_rollouts': n_rollouts, 'test_start_seed': test_start_seed})

    results = {'ckpt': ckpt, 'method': method, 'n_rollouts': n_rollouts,
               'test_start_seed': test_start_seed, 'conditions': {}}
    jsonl_path = os.path.join(output_dir, 'per_rollout_steps.jsonl')
    for name, z in conditions.items():
        print(f"\n=== {method} @ {name} = {z} ({n_rollouts} rollouts) ===")
        runner.target_rewards = np.array(z, dtype=np.float32)
        log = runner.run(policy)
        # the runner appends one line per run() with the per-rollout steps
        with open(jsonl_path) as f:
            steps = json.loads(f.readlines()[-1])['prefixes']['test/']
        summary = rollout_summary(steps['first_success_step_left'],
                                  steps['first_success_step_right'], n_rollouts)
        runner_metrics = {k[len('test/'):]: float(v) for k, v in log.items()
                          if k.startswith('test/') and not k.startswith('test/sim_')
                          and isinstance(v, (int, float, np.floating, np.integer))}
        results['conditions'][name] = {
            'target': z,
            'summary': summary,
            'runner_metrics': runner_metrics,
            'first_success_step_left': steps['first_success_step_left'],
            'first_success_step_right': steps['first_success_step_right'],
        }
        print(json.dumps(summary, indent=2))
        wandb.log({**{f'{name}/{k}': v for k, v in summary.items() if v is not None},
                   **{f'{name}/video_{k[len("test/sim_video_"):]}': v for k, v in log.items()
                      if k.startswith('test/sim_video_')}})

    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    wandb.finish()
    runner.env.close()


if __name__ == '__main__':
    main()
