"""
Roll out a goal-conditioned policy on the two-peg task: evaluate it and,
optionally, record its episodes as training data (filtered BC / AWR / UDRL).

Each peg is evaluated on the same env seeds (same nut start positions), so
task A and task B results are paired. Success means the nut lands on the
*commanded* peg; time to success is the first step it is on that peg.

Usage:
  python -m transfer.rollout --ckpt final.ckpt --pegs left,right --n 100 \
      --seed_start 300000 --out_json eval.json [--record rollouts.hdf5]
"""

import json
import os
import time

import click
import numpy as np
import torch

from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv

from transfer.common import (MAX_STEPS, TASK_PEG, TASKS, env_meta_from, goal_vec, load_checkpoint, make_env_fn,
                             to_env_action, write_episodes)

PEG_NAMES = {0: 'none', 1: 'left', 2: 'right'}
LIFT_HEIGHT = 0.05      # m above its start: the nut was grasped and lifted


def summarize(episodes):
    n = len(episodes)
    succ = [e for e in episodes if e['success']]
    t1s = [e['first_success_step'] for e in succ]
    lifted = [e for e in episodes if e['lifted']]
    return {
        'n': n,
        'success': len(succ) / n,
        'grasp': len(lifted) / n,                                       # nut lifted off the table at some point
        'place_given_grasp': (sum(e['success'] for e in lifted) / len(lifted)) if lifted else None,
        'wrong_peg': sum(e['final_peg'] not in ('none', TASK_PEG[e['target_peg']]) for e in episodes) / n,
        't1s': float(np.mean(t1s)) if t1s else None,
        'throughput': float(sum(300.0 / t for t in t1s) / n),
    }


@click.command()
@click.option('--ckpt', required=True)
@click.option('--pegs', default='left,right')
@click.option('--n', default=100, type=int, help='Episodes per peg')
@click.option('--seed_start', default=300000, type=int)
@click.option('--n_envs', default=None, type=int, help='Default: CPUs available - 2')
@click.option('--env_meta_from', 'env_meta_path', required=True, help='hdf5 file whose env_args to use')
@click.option('--out_json', required=True)
@click.option('--record', default=None, help='Write episodes (hdf5) for training')
@click.option('--cond', default=None, help='Extra conditioning (JSON list); default: the checkpoint\'s eval_cond')
@click.option('--nut_range', default=None, help='JSON [[xlo, ylo], [xhi, yhi]] nut start range (default: training range)')
@click.option('--no_ema', is_flag=True, default=False)
@click.option('--sigma', default=0.0, type=float, help='Sample with PPO\'s exploration noise (0: deterministic)')
@click.option('--noise_steps', default='0,6', help='[start,end) of the noisy integration steps')
@click.option('--push_step', default=None, type=int, help='Robustness probe: env step at which the nut is moved')
@click.option('--push_dist', default=0.04, type=float)
@click.option('--friction_scale', default=1.0, type=float, help='Robustness probe: scale the nut\'s friction')
@click.option('--peg2_shift', default=None, help='Probe: move the right (round) peg by "dx,dy" meters')
def main(ckpt, pegs, n, seed_start, n_envs, env_meta_path, out_json, record, cond, nut_range, no_ema, sigma, noise_steps,
         push_step, push_dist, friction_scale, peg2_shift):
    t0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy, ema_policy, payload = load_checkpoint(ckpt, device)
    policy = policy if no_ema else ema_policy
    policy.eval()
    extra = json.loads(cond) if cond else payload['info'].get('eval_cond', [])
    assert len(extra) == payload['extra_cond_dim'], (extra, payload['extra_cond_dim'])
    nut_range = json.loads(nut_range) if nut_range else None
    peg2_shift = [float(x) for x in peg2_shift.split(',')] if peg2_shift else None
    actor = None
    if sigma > 0:
        from transfer.ppo import Actor
        actor = Actor(policy, sigma, [int(x) for x in noise_steps.split(',')])

    if n_envs is None:
        n_envs = max(1, len(os.sched_getaffinity(0)) - 2)
    n_envs = min(n_envs, n)
    env_meta = env_meta_from(env_meta_path)
    envs = {}

    def env_for(peg):
        nut = 'round' if peg == 'round' else 'square'
        if nut not in envs:
            envs[nut] = AsyncVectorEnv([make_env_fn(env_meta, nut_range=nut_range, push_step=push_step, push_dist=push_dist,
                                                    friction_scale=friction_scale, nut=nut, peg2_shift=peg2_shift)] * n_envs)
        return envs[nut]

    results, recorded = {}, []
    for peg in pegs.split(','):
        assert peg in TASKS
        env = env_for(peg)
        cond_vec = np.concatenate([goal_vec(peg), np.asarray(extra, dtype=np.float32)]).astype(np.float32)
        episodes = []
        seeds = list(range(seed_start, seed_start + n))
        for start in range(0, n, n_envs):
            chunk = seeds[start:start + n_envs]
            k = len(chunk)
            env.seed(chunk + [chunk[0]] * (n_envs - k))
            obs = env.reset()
            policy.reset()
            succeeded = np.zeros(n_envs, dtype=bool)
            for _ in range(int(np.ceil(MAX_STEPS / policy.n_action_steps))):
                o = np.concatenate([obs, np.broadcast_to(cond_vec, obs.shape[:2] + cond_vec.shape)], axis=-1)
                with torch.no_grad():
                    o_t = torch.from_numpy(o.astype(np.float32)).to(device)
                    action = actor.sample(actor.cond(o_t))[0] if actor else policy.predict_action({'obs': o_t})['action']
                obs, reward, done, info = env.step(to_env_action(action.cpu().numpy()))
                succeeded |= np.asarray(reward) >= 1.0
                if succeeded[:k].all() or np.all(done):
                    break
            for i, traj in enumerate(env.call('get_trajectory')[:k]):
                pegs_t = np.asarray(traj['peg'])
                on = np.nonzero(pegs_t)[0]
                first = int(on[0]) if len(on) else -1
                final_peg = PEG_NAMES[int(pegs_t[first])] if len(on) else 'none'
                T = first + 1 if len(on) else len(traj['action7'])
                nut_z = np.asarray(traj['obs'])[:, 2]                    # obs starts with the nut position
                ep = {'target_peg': peg, 'seed': chunk[i], 'final_peg': final_peg,
                      'success': final_peg == TASK_PEG[peg], 'first_success_step': first if final_peg == TASK_PEG[peg] else -1,
                      'lifted': bool(nut_z.max() > nut_z[0] + LIFT_HEIGHT)}
                episodes.append(ep)
                if record:
                    recorded.append({'obs': np.asarray(traj['obs'][:T], dtype=np.float32),
                                     'action7': np.asarray(traj['action7'][:T], dtype=np.float32),
                                     'attrs': {**ep, 'success': bool(ep['success'])}})
            print(f"  {peg}: {len(episodes)}/{n} episodes, success so far "
                  f"{np.mean([e['success'] for e in episodes]):.2f}", flush=True)
        results[peg] = {**summarize(episodes), 'episodes': episodes}
        print(f"{peg}: {json.dumps({k: v for k, v in results[peg].items() if k != 'episodes'})}", flush=True)
    for e in envs.values():
        e.close()

    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump({'ckpt': ckpt, 'cond': extra, 'nut_range': nut_range, 'seed_start': seed_start,
                   'ema': not no_ema, 'sigma': sigma, 'noise_steps': noise_steps, 'push_step': push_step,
                   'push_dist': push_dist if push_step is not None else None, 'friction_scale': friction_scale,
                   'peg2_shift': peg2_shift,
                   'results': results}, f, indent=1)
    if record:
        write_episodes(record, recorded, env_meta)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out_json}" + (f", {record}" if record else ''))


if __name__ == '__main__':
    main()
