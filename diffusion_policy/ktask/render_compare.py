"""
Side-by-side videos of several policies on the same start states: one cell per task, the policies next to
each other inside it, a green border once an episode has succeeded. Needs a Vulkan GPU (iris-ws-4).

  VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json python -m ktask.render_compare -o out.mp4 \
      --policy "base=/path/base.ckpt" --policy "PPO on 1 task=/path/ppo.ckpt" --tasks 13,12,8,0 [--cols 2]
"""

import click
import gymnasium as gym
import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

import ktask.env  # noqa: F401
from ktask.data import N_ACT, SPLITS, parse_tasks
from ktask.env import task_name, to_env_action
from ktask.policy import load_checkpoint

SPLIT_OF = {t: s for s, ts in SPLITS.items() for t in ts}


def caption(img, lines, ok):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 13 * len(lines) + 2], fill=(255, 255, 255))
    for i, l in enumerate(lines):
        d.text((4, 1 + 13 * i), l, fill=(0, 0, 0))
    a = np.asarray(im).copy()
    if ok:
        a[:, :5], a[:, -5:], a[-5:] = [30, 200, 30], [30, 200, 30], [30, 200, 30]
    return a


@click.command()
@click.option('-o', '--out', required=True)
@click.option('--policy', 'policies', multiple=True, required=True, help='label=checkpoint')
@click.option('--tasks', required=True)
@click.option('--episode', default=0, help='which evaluation start state (same for every policy)')
@click.option('--cols', default=2, help='task cells per row')
@click.option('--steps', default=200)
@click.option('--size', default=224)
@click.option('--seed', default=300_000)
def main(out, policies, tasks, episode, cols, steps, size, seed):
    tasks = parse_tasks(tasks) if not tasks.replace(',', '').isdigit() else [int(t) for t in tasks.split(',')]
    pols = [(p.split('=', 1)[0], p.split('=', 1)[1]) for p in policies]
    P = len(pols)
    ids = [t for t in tasks for _ in range(P)]                 # slot i: task i // P, policy i % P
    env = gym.make('KTaskPlace-v0', num_envs=len(ids), task_ids=ids, obs_mode='state', control_mode='pd_ee_pose',
                   sim_backend='gpu', render_mode='rgb_array', human_render_camera_configs=dict(width=size, height=size))
    u = env.unwrapped
    env.reset(seed=[seed + 1000 * t + episode for t in ids])  # the evaluation start states, paired across policies
    dev = u.device
    nets = [load_checkpoint(path, dev)[0].eval() for _, path in pols]
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    o = u.policy_obs()
    hist = torch.stack([o, o], 1)
    done_at = np.full(len(ids), -1)
    chunk = torch.zeros(len(ids), N_ACT, 10, device=dev)
    frames = []
    for t in range(steps):
        if t % N_ACT == 0:
            with torch.no_grad():
                for j, net in enumerate(nets):
                    idx = torch.arange(j, len(ids), P, device=dev)
                    chunk[idx] = net.sample(hist[idx], gen)
        if t % N_ACT == N_ACT - 1:
            prev = u.policy_obs()
        _, _, _, _, info = env.step(to_env_action(chunk[:, t % N_ACT]))
        s = info['success'].cpu().numpy()
        done_at[(done_at < 0) & s] = t
        if t % N_ACT == N_ACT - 1:
            o = u.policy_obs()
            hist = torch.stack([prev, o], 1)                    # consecutive env steps, as in training
        if t % 2 == 0:
            imgs = env.render().cpu().numpy()
            cells = []
            for i, task in enumerate(tasks):
                row = [caption(imgs[i * P + j], [f'{task_name(task)} [{SPLIT_OF[task]}]', lab],
                               done_at[i * P + j] >= 0) for j, (lab, _) in enumerate(pols)]
                cells.append(np.concatenate(row, 1))
            while len(cells) % cols:
                cells.append(np.full_like(cells[0], 255))
            frames.append(np.concatenate([np.concatenate(cells[r:r + cols], 1) for r in range(0, len(cells), cols)], 0))
    imageio.mimsave(out, frames, fps=10)
    for j, (lab, _) in enumerate(pols):
        print(f"{lab}: success on {int(np.sum(done_at[j::P] >= 0))}/{len(tasks)} tasks")
    print(f'saved {out}')


if __name__ == '__main__':
    main()
