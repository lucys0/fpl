"""
Video of every task side by side (8 objects as rows x 4 targets as columns), driven by the scripted
operator (--q) or a policy checkpoint (--ckpt). Needs a Vulkan-capable GPU (iris-ws-4).

  VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json python -m ktask.render -o tasks.mp4 [--q 0] [--ckpt C]
"""

import click
import gymnasium as gym
import imageio
import numpy as np
import torch

import ktask.env  # noqa: F401
from ktask.data import N_ACT
from ktask.env import N_OBJ, N_TASKS, N_TGT, OBJ_NAMES, TGT_NAMES, to_env_action
from ktask.expert import Expert


def label(img, text):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    ImageDraw.Draw(im).text((6, 4), text, fill=(0, 0, 0))
    return np.asarray(im)


@click.command()
@click.option('-o', '--out', required=True)
@click.option('--q', default=0.0)
@click.option('--ckpt', default=None)
@click.option('--steps', default=160)
@click.option('--seed', default=0)
@click.option('--size', default=192)
def main(out, q, ckpt, steps, seed, size):
    ids = list(range(N_TASKS))
    env = gym.make('KTaskPlace-v0', num_envs=N_TASKS, task_ids=ids, obs_mode='state', control_mode='pd_ee_pose',
                   sim_backend='gpu', render_mode='rgb_array', human_render_camera_configs=dict(width=size, height=size))
    u = env.unwrapped
    env.reset(seed=[seed + i for i in ids])
    gen = torch.Generator(device=u.device)
    gen.manual_seed(seed)
    if ckpt:
        from ktask.policy import load_checkpoint
        policy, _ = load_checkpoint(ckpt, u.device)
        policy.eval()
        o = u.policy_obs()
        hist = torch.stack([o, o], 1)
        chunk, j = None, N_ACT
    else:
        ex = Expert(u, torch.full((N_TASKS,), q), gen)
    done_at = torch.full((N_TASKS,), -1)
    frames = []
    for t in range(steps):
        if ckpt:
            if j == N_ACT:
                chunk, j = policy.sample(hist, gen), 0
            a = chunk[:, j]
            j += 1
        else:
            a = ex.act()
        _, _, _, _, info = env.step(to_env_action(a))
        s = info['success'].cpu()
        done_at[(done_at < 0) & s] = t
        if ckpt and j == N_ACT:
            o = u.policy_obs()
            hist = torch.stack([hist[:, 1], o], 1)
        if t % 2 == 0:
            imgs = env.render().cpu().numpy()                      # (N, H, W, 3)
            tiles = []
            for i in range(N_TASKS):
                img = imgs[i].copy()
                if done_at[i] >= 0:
                    img[:6] = [40, 200, 40]
                    img[-6:] = [40, 200, 40]
                tiles.append(label(img, f'{OBJ_NAMES[i // N_TGT]}-{TGT_NAMES[i % N_TGT]}'))
            grid = np.concatenate([np.concatenate(tiles[r * N_TGT:(r + 1) * N_TGT], 1) for r in range(N_OBJ)], 0)
            frames.append(grid)
    imageio.mimsave(out, frames, fps=10)
    print(f'saved {out}: {len(frames)} frames, successes {(done_at >= 0).float().mean():.2f}')


if __name__ == '__main__':
    main()
