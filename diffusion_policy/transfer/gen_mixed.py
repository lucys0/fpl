"""
Mixed-quality scripted trajectories: an imperfect "operator" that misjudges the
grasp and the placement, recorded once per episode whether it succeeds or not.

The authors' scripted policy (scripts/collect_initial_scripted_rollouts.py)
plans waypoints from the true nut and peg poses and retries until it succeeds.
Here each episode draws a quality level q ~ U(0, q_max) and offsets the plan:
  - grasp phase (approach, descend, close, lift): xy shift ~ N(0, (q*sigma_grasp)^2), yaw ~ N(0, (q*sigma_yaw)^2)
  - placement phase (insert, release, hold): xy shift ~ N(0, (q*sigma_place)^2)
  - transit between them: the offset blends linearly from the grasp shift to the placement shift
Every attempt is kept with attrs target_peg, success, quality, the offsets, and
first_success_step. Episodes run at most --max_steps (default 500, the eval limit).

Usage:
  python -m transfer.gen_mixed -o OUT.hdf5 -n 25 --seed 1 --target_peg left [--q_max 1 --sigma_grasp 0.03 ...]
"""

import importlib.util
import json

import click
import h5py
import numpy as np

REPO = '/iris/u/lucyshi/fpl/diffusion_policy'

# indices into SquareSideScriptedPolicy.trajectory (see its generate_trajectory)
GRASP = range(0, 62)       # above nut, descend (30), close gripper, lift (30)
ROTATE = 62
TRANSIT = range(63, 93)    # 30 points
PLACE_START = 93           # insert (25), release, hold


def load_collector():
    spec = importlib.util.spec_from_file_location('collector', f'{REPO}/scripts/collect_initial_scripted_rollouts.py')
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)          # changes into the repo dir, as the script does
    return collector


def perturb(trajectory, d_grasp, d_yaw, d_place):
    assert len(trajectory) == 120, len(trajectory)
    for i in GRASP:
        trajectory[i]['action'][:2] += d_grasp
        trajectory[i]['action'][5] += d_yaw
    trajectory[ROTATE]['action'][:2] += d_grasp
    for j, i in enumerate(TRANSIT):
        w = (j + 1) / (len(TRANSIT) + 1)
        trajectory[i]['action'][:2] += (1 - w) * d_grasp + w * d_place
    for i in range(PLACE_START, len(trajectory)):
        trajectory[i]['action'][:2] += d_place


@click.command()
@click.option('-o', '--out', required=True)
@click.option('-n', '--num_episodes', type=int, required=True)
@click.option('--seed', type=int, required=True)
@click.option('--target_peg', type=click.Choice(['left', 'right', 'round']), required=True,
              help='round: the round nut onto the right peg (round_task.py)')
@click.option('--q_min', type=float, default=0.0)
@click.option('--q_max', type=float, default=1.0)
@click.option('--sigma_grasp', type=float, default=0.03, help='m, at q = 1')
@click.option('--sigma_yaw', type=float, default=0.3, help='rad, at q = 1')
@click.option('--sigma_place', type=float, default=0.02, help='m, at q = 1')
@click.option('--max_steps', type=int, default=500)
def main(out, num_episodes, seed, target_peg, q_min, q_max, sigma_grasp, sigma_yaw, sigma_place, max_steps):
    collector = load_collector()
    np.random.seed(seed)
    collector.random.seed(seed)
    if target_peg == 'round':
        from transfer.round_task import collector_round_classes, create_round_collector_env
        env, env_cfg = create_round_collector_env(collector)
        RoundPolicy = collector_round_classes(collector)[1]
        make_policy = lambda **kw: RoundPolicy(env, target_peg='right', **kw)
    else:
        env, env_cfg = collector.create_robomimic_env()
        make_policy = lambda **kw: collector.create_policy(env, target_peg=target_peg, **kw)
    with h5py.File(out, 'w') as f:
        data = f.create_group('data')
        data.attrs['env_args'] = json.dumps(env_cfg)
        n_succ = 0
        for ep in range(num_episodes):
            policy = make_policy(noise_level=np.random.uniform(0, 0.12), speed_factor=1.0)
            policy.speed_factor = np.random.uniform(1.0, 2.0)
            q = np.random.uniform(q_min, q_max)
            d_grasp = np.random.normal(0, q * sigma_grasp, 2)
            d_yaw = np.random.normal(0, q * sigma_yaw)
            d_place = np.random.normal(0, q * sigma_place, 2)
            env.env.env.env.init_state = None
            env.seed(np.random.randint(0, 10_000_000))
            obs = env.reset()
            policy.replan()
            perturb(policy.trajectory, d_grasp, d_yaw, d_place)
            obs_list, action_list, success, first = [], [], False, -1
            for step in range(max_steps):
                action = policy.predict_action(obs)
                obs_list.append(obs[-1].copy())
                action_list.append(action[0].copy())
                obs, reward, done, info = env.step(action)
                if reward >= 1.0:
                    success, first = True, step
                    break
            n_succ += success
            g = data.create_group(f'demo_{ep}')
            obs_g = g.create_group('obs')
            arr = np.stack(obs_list)
            idx = 0
            for key, dim in zip(collector.OBS_KEYS, (14, 3, 4, 2)):
                obs_g.create_dataset(key, data=arr[:, idx:idx + dim])
                idx += dim
            g.create_dataset('actions', data=np.stack(action_list))
            g.create_dataset('states', data=np.broadcast_to(env.env.env.last_reset_state,
                                                           (len(obs_list), len(env.env.env.last_reset_state))).copy())
            for k, v in dict(target_peg=target_peg, success=success, first_success_step=first, quality=q,
                             d_grasp=d_grasp, d_yaw=d_yaw, d_place=d_place, speed_factor=policy.speed_factor).items():
                g.attrs[k] = v
            print(f"episode {ep}: q={q:.2f} grasp {np.linalg.norm(d_grasp) * 100:.1f}cm yaw {abs(d_yaw):.2f} "
                  f"place {np.linalg.norm(d_place) * 100:.1f}cm -> {'success' if success else 'fail'}", flush=True)
        data.attrs['success_rate'] = n_succ / num_episodes
    print(f"{out}: {n_succ}/{num_episodes} successful")


if __name__ == '__main__':
    main()
