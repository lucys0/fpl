"""
Videos of rollouts (two cameras side by side, captioned), one MP4 per episode, named by outcome.

  python -m transfer.render_videos --policy CKPT --task round --label base_s2 -n 8 --out_dir DIR
  python -m transfer.render_videos --policy scripted --task round --q 0.0 --label scripted_clean -n 2 --out_dir DIR

Policies roll out from the evaluation start states (seed 300000 + i), as transfer.rollout does;
the scripted round-nut policy (ring grasp, round_task.py) is perturbed at quality q as in gen_mixed.py.
"""

import math
import os

import av
import click
import numpy as np
import torch
import matplotlib
from PIL import Image, ImageDraw, ImageFont

from transfer.common import (MAX_STEPS, N_ACTION_STEPS, N_OBS_STEPS, OBS_KEYS, RecordingWrapper, TASK_PEG,
                             env_meta_from, goal_vec, load_checkpoint, to_env_action)

CAMERAS = ('agentview', 'robot0_eye_in_hand')     # scene (facing the robot: A peg on the right) + wrist camera
PEG_NAMES = {0: 'none', 1: 'left', 2: 'right'}


def grab(rob_env, size):
    """Frames from CAMERAS side by side; rob_env is robomimic's EnvRobosuite."""
    return np.concatenate([rob_env.render(mode='rgb_array', height=size, width=size, camera_name=c) for c in CAMERAS],
                          axis=1)


FONT = ImageFont.truetype(os.path.join(matplotlib.get_data_path(), 'fonts/ttf/DejaVuSans.ttf'), 13)
TASK_TEXT = {'round': 'B: round nut, grasp the ring, onto the round peg (grey)',
             'left': 'A: square nut, grasp the handle, onto the square peg (wood)',
             'right': 'B: square nut onto the round peg (grey)'}


def caption(frame, lines):
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im)
    for i, line in enumerate(lines):
        w = d.textlength(line, font=FONT)
        d.rectangle([0, 17 * i, w + 8, 17 * i + 17], fill=(0, 0, 0))
        d.text((4, 17 * i + 1), line, fill=(255, 255, 255), font=FONT)
    return np.asarray(im)


def write_mp4(path, frames, fps=10):
    c = av.open(path, mode='w')
    s = c.add_stream('libx264', rate=fps)
    s.height, s.width = frames[0].shape[:2]
    s.pix_fmt = 'yuv420p'
    s.options = {'crf': '23'}
    for f in frames:
        for p in s.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format='rgb24')):
            c.mux(p)
    for p in s.encode():
        c.mux(p)
    c.close()


@click.command()
@click.option('--policy', required=True, help="checkpoint path, or 'scripted' (round nut, ring grasp)")
@click.option('--task', type=click.Choice(['left', 'right', 'round']), default='round')
@click.option('--label', required=True)
@click.option('-n', '--num_episodes', default=8)
@click.option('--seed_start', default=300000)
@click.option('--q', default=0.0, help='scripted only: demo quality level (0 = clean)')
@click.option('--every', default=2, help='render every N env steps (20 Hz control)')
@click.option('--size', default=320)
@click.option('--env_meta_from', 'env_meta_path', default='/iris/u/lucyshi/fpl_transfer/data/pretrain_left.hdf5')
@click.option('--out_dir', required=True)
def main(policy, task, label, num_episodes, seed_start, q, every, size, env_meta_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    if policy == 'scripted':
        return scripted(task, label, num_episodes, seed_start, q, every, size, out_dir)
    import gym
    from diffusion_policy.env.robomimic.robomimic_twopeg_lowdim_wrapper import RobomimicTwoPegLowdimWrapper
    from diffusion_policy.env_runner.robomimic_lowdim_runner import create_env
    from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
    from transfer.round_task import RoundNutTwoPegWrapper, round_env_meta

    env_meta = env_meta_from(env_meta_path)
    if task == 'round':
        twopeg = RoundNutTwoPegWrapper(env=create_env(env_meta=round_env_meta(env_meta), obs_keys=OBS_KEYS),
                                       obs_keys=OBS_KEYS, init_state=None)
    else:
        twopeg = RobomimicTwoPegLowdimWrapper(env=create_env(env_meta=env_meta, obs_keys=OBS_KEYS), obs_keys=OBS_KEYS,
                                              init_state=None)

    class Grab(gym.Wrapper):
        def reset(self, **kw):
            obs = self.env.reset(**kw)
            self.t, self.frames = 0, [grab(twopeg.env, size)]
            return obs

        def step(self, a):
            out = self.env.step(a)
            self.t += 1
            if self.t % every == 0:
                self.frames.append(grab(twopeg.env, size))
            return out

    grabber = Grab(twopeg)
    rec = RecordingWrapper(grabber)
    env = MultiStepWrapper(rec, n_obs_steps=N_OBS_STEPS, n_action_steps=N_ACTION_STEPS, max_episode_steps=MAX_STEPS)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, pol, payload = load_checkpoint(policy, device)
    pol.eval()
    goal = goal_vec(task)
    for i in range(num_episodes):
        seed = seed_start + i
        torch.manual_seed(seed)              # the policy samples from noise: fix it so videos are reproducible
        env.seed(seed)
        obs = env.reset()
        pol.reset() if hasattr(pol, 'reset') else None
        ended_at = None
        for _ in range(math.ceil(MAX_STEPS / N_ACTION_STEPS)):
            o = np.concatenate([obs, np.broadcast_to(goal, obs.shape[:1] + goal.shape)], axis=-1)[None].astype(np.float32)
            with torch.no_grad():
                a = pol.predict_action({'obs': torch.from_numpy(o).to(device)})['action'].cpu().numpy()
            obs, _, done, _ = env.step(to_env_action(a)[0])
            peg, _ = rec.get_first_peg()
            if peg and ended_at is None:
                ended_at = len(grabber.frames)
            if (ended_at is not None and len(grabber.frames) > ended_at + 15) or done:
                break
        peg, step = rec.get_first_peg()
        final = PEG_NAMES[peg]
        ok = final == TASK_PEG[task]
        outcome = 'SUCCESS' if ok else ('WRONG PEG' if final != 'none' else 'FAIL')
        frames = [caption(f, [f'{label} | {TASK_TEXT[task]}',
                              f'start {seed} | step {min(k * every, MAX_STEPS)} | episode: {outcome}'
                              + (f' (on peg at step {step})' if ok else '')])
                  for k, f in enumerate(grabber.frames)]
        path = f"{out_dir}/{label}_{task}_seed{seed}_{'success' if ok else 'fail'}.mp4"
        write_mp4(path, frames)
        print(f'{path}: {outcome}, {len(frames)} frames', flush=True)


def scripted(task, label, num_episodes, seed_start, q, every, size, out_dir):
    """Scripted round-nut demos at quality q (gen_mixed.py's perturbation), rendered."""
    assert task == 'round'
    from transfer.gen_mixed import load_collector, perturb
    from transfer.round_task import collector_round_classes, create_round_collector_env
    collector = load_collector()
    np.random.seed(seed_start)
    collector.random.seed(seed_start)
    env, _ = create_round_collector_env(collector)
    Policy = collector_round_classes(collector)[1]
    rob = env.env.env.env.env           # MultiStep -> Video -> RoundMultimodal -> RobomimicLowdim -> EnvRobosuite
    for i in range(num_episodes):
        policy = Policy(env, target_peg='right', noise_level=np.random.uniform(0, 0.12), speed_factor=1.0)
        policy.speed_factor = np.random.uniform(1.0, 2.0)
        d_grasp, d_yaw, d_place = (np.random.normal(0, q * 0.03, 2), np.random.normal(0, q * 0.3),
                                   np.random.normal(0, q * 0.02, 2))
        env.env.env.env.init_state = None
        env.seed(np.random.randint(0, 10_000_000))
        obs = env.reset()
        policy.replan()
        perturb(policy.trajectory, d_grasp, d_yaw, d_place)
        frames, ok, end = [grab(rob, size)], False, None
        for t in range(MAX_STEPS):
            obs, reward, done, info = env.step(policy.predict_action(obs))
            if (t + 1) % every == 0:
                frames.append(grab(rob, size))
            if reward >= 1.0 and end is None:
                ok, end = True, t
            if end is not None and t > end + 30:
                break
        info_line = f'grasp offset {np.linalg.norm(d_grasp) * 100:.1f} cm, yaw {abs(d_yaw):.2f}, place offset {np.linalg.norm(d_place) * 100:.1f} cm'
        frames = [caption(f, [f'{label}: scripted demo, quality q={q} | {TASK_TEXT["round"]}',
                              f'{info_line} | episode: {"SUCCESS" if ok else "FAIL"}']) for f in frames]
        path = f"{out_dir}/{label}_round_ep{i}_{'success' if ok else 'fail'}.mp4"
        write_mp4(path, frames)
        print(f'{path}: {"SUCCESS" if ok else "FAIL"}, {len(frames)} frames', flush=True)


if __name__ == '__main__':
    main()
