"""
Where and how post-training on A changes the policy, on A vs B (seed 0).

Figure 1 (trajectories): top-down end-effector paths of recorded rollouts with goal A and goal B,
for the base, PPO (6000 episodes), SFT on expert demos at PPO's KL from the base (lr 1e-6, 700
steps) and SFT after B collapsed (3000 steps).

Figure 2 (action changes), on the states the base visits in its own rollouts (each with its goal):
  maps   - grid cells over the end-effector position, colored by the mean change of the planned
           positions vs the base (cm), arrows = mean change direction (x, y)
  phases - mean change per task phase and goal, for checkpoints along training
  noise  - KL from the base per denoising step (0 = pure noise ... 9 = final), A vs B states

  python -m transfer.m_viz [--out_dir results/viz]
"""

import os

import click
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch

from transfer.common import ACTION_DIM, HORIZON, N_OBS_STEPS, OBS_KEYS, goal_vec, load_checkpoint
from transfer.kl import A0, A1, SIGMA, cond_of, ref_chain

R = '/iris/u/lucyshi/fpl_transfer/runs'
PEGS_XY = {'left': (0.23, 0.10), 'right': (0.23, -0.10)}
EEF = slice(14, 17)            # robot0_eef_pos in the 23-D observation
NUT = slice(0, 3)              # SquareNut_pos
NUT_TO_EEF = slice(7, 10)      # SquareNut_to_robot0_eef_pos
PHASES = ('reach', 'grasp', 'transport', 'insert')

CKPTS = {   # label -> checkpoint (seed 0)
    'PPO 1000 ep.': f'{R}/m2/s0/ppo/ppo/ppo_ep1000.ckpt',
    'PPO 3000 ep.': f'{R}/m2/s0/ppo/ppo/ppo_ep3000.ckpt',
    'PPO 6000 ep.': f'{R}/m2/s0/ppo/ppo/ppo_ep6000.ckpt',
    'SFT lr1e-6 200': f'{R}/m5/s0/sft_expert_lr1e-06early/train/step200.ckpt',
    'SFT lr1e-6 400': f'{R}/m5/s0/sft_expert_lr1e-06early/train/step400.ckpt',
    'SFT lr1e-6 700': f'{R}/m5/s0/sft_expert_lr1e-06early/train/step700.ckpt',
    'SFT lr1e-6 3000': f'{R}/m5/s0/sft_expert_lr1e-06/train/step3000.ckpt',
    'filtered BC lr1e-6 400': f'{R}/m5/s0/filtered_bc_lr1e-06early/train/step400.ckpt',
    'filtered BC lr1e-6 1000': f'{R}/m5/s0/filtered_bc_lr1e-06early/train/final.ckpt',
    'SFT expert N=1000, replay': f'{R}/m1r/s0/sft_expert/n1000/final.ckpt',
    'AWR N=1000, no replay': f'{R}/m1/s0/awr/n1000/final.ckpt',
}
MAP_CKPTS = ('PPO 6000 ep.', 'SFT lr1e-6 700')       # matched KL from the base on A states (4.9 vs 5.4)
MAP_TITLES = {'PPO 6000 ep.': 'PPO, 6000 episodes',
              'SFT lr1e-6 700': 'SFT on expert A demos, 700 steps\n(same KL from the base as PPO)'}


def load_rollouts(path):
    eps = []
    with h5py.File(path, 'r') as f:
        for k in sorted(f['data'].keys(), key=lambda k: int(k.split('_')[1])):
            d = f['data'][k]
            obs = np.concatenate([d['obs'][key][:] for key in OBS_KEYS], axis=-1).astype(np.float32)
            eps.append({'obs': obs, 'peg': d.attrs['target_peg'] if isinstance(d.attrs['target_peg'], str)
                        else d.attrs['target_peg'].decode(), 'success': bool(d.attrs['success'])})
    return eps


def phase_of(obs, peg):
    """Per-step phase from the observation: reach / grasp (nut not lifted yet; gripper more / less than
    9 cm from the nut's center - it grasps the handle 7 cm out), transport (lifted, more than 5 cm from the
    target peg), insert (within 5 cm of it)."""
    z0 = obs[0, NUT][2]
    lifted = np.maximum.accumulate(obs[:, NUT][:, 2] > z0 + 0.03)
    near_nut = np.linalg.norm(obs[:, NUT_TO_EEF], axis=1) < 0.09
    near_peg = np.linalg.norm(obs[:, NUT][:, :2] - np.array(PEGS_XY[peg]), axis=1) < 0.05
    return np.where(~lifted, np.where(near_nut, 1, 0), np.where(near_peg, 3, 2))


def windows(eps, peg, max_states):
    """Observation windows (with the goal), eef positions and phases from the base's rollouts."""
    obs_w, eef, ph = [], [], []
    for ep in eps:
        if ep['peg'] != peg:
            continue
        o, p = ep['obs'], phase_of(ep['obs'], peg)
        for t in range(N_OBS_STEPS - 1, len(o)):
            obs_w.append(o[t - N_OBS_STEPS + 1:t + 1]); eef.append(o[t, EEF]); ph.append(p[t])
    idx = np.random.default_rng(0).permutation(len(obs_w))[:max_states]
    g = goal_vec(peg)
    w = np.stack(obs_w)[idx]
    w = np.concatenate([w, np.broadcast_to(g, w.shape[:2] + g.shape)], axis=-1).astype(np.float32)
    return w, np.stack(eef)[idx], np.array(ph)[idx]


@torch.no_grad()
def planned(policy, w, device, bs=1000):
    out = []
    for i in range(0, len(w), bs):
        torch.manual_seed(1000 + i)
        out.append(policy.predict_action({'obs': torch.from_numpy(w[i:i + bs]).to(device)})['action'][..., :3]
                   .cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def kl_per_step(pol, base, w, device, bs=500):
    """KL(base || pol) per denoising step, averaged over states: (K,)."""
    tot = None
    for i in range(0, len(w), bs):
        torch.manual_seed(2000 + i)
        cond = cond_of(base, torch.from_numpy(w[i:i + bs]).to(device))
        xs, ts, mus = ref_chain(base, cond)
        B, K = ts.shape
        x = xs.reshape(B * K, HORIZON, ACTION_DIM)
        mu = x + (-1.0 / K) * pol.model(x, ts.reshape(B * K), cond.repeat_interleave(K, 0))
        d = (mu.view(B, K, HORIZON, ACTION_DIM) - mus)[:, :, A0:A1]
        s = (d.pow(2).sum(dim=(2, 3)) / (2 * SIGMA ** 2)).sum(0).cpu().numpy()
        tot = s if tot is None else tot + s
    return tot / len(w)


def draw_scene(ax):
    ax.add_patch(plt.Rectangle((PEGS_XY['left'][0] - 0.016, PEGS_XY['left'][1] - 0.016), 0.032, 0.032,
                               color='k', alpha=0.7, zorder=5))
    ax.add_patch(plt.Circle(PEGS_XY['right'], 0.02, color='k', alpha=0.7, zorder=5))
    ax.text(PEGS_XY['left'][0] + 0.03, PEGS_XY['left'][1], 'A peg', fontsize=7, va='center')
    ax.text(PEGS_XY['right'][0] + 0.03, PEGS_XY['right'][1], 'B peg', fontsize=7, va='center')
    ax.add_patch(plt.Rectangle((-0.2, -0.2), 0.2, 0.4, fill=False, ls=':', color='gray', lw=0.8))
    ax.set_xlim(-0.28, 0.33); ax.set_ylim(-0.27, 0.27); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])


@click.command()
@click.option('--out_dir', default='/iris/u/lucyshi/fpl_transfer/results/viz')
@click.option('--max_states', default=6000)
def main(out_dir, max_states):
    device = torch.device('cuda')
    roll = {name: load_rollouts(f'{out_dir}/{name}.hdf5') for name in ('base', 'ppo6000', 'sft700', 'sft3000')}

    # ---------------- figure 1: trajectories
    titles = {'base': 'base', 'ppo6000': 'PPO, 6000 episodes',
              'sft700': 'SFT on expert A demos, lr 1e-6, 700 steps\n(moved as far from the base as PPO: KL 5.4 vs 4.9)',
              'sft3000': 'SFT on expert A demos, lr 1e-6, 3000 steps\n(KL 37)'}
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.6))
    for j, (name, eps) in enumerate(roll.items()):
        for i, peg in enumerate(('left', 'right')):
            ax = axes[i, j]
            draw_scene(ax)
            sel = [e for e in eps if e['peg'] == peg][:20]
            for e in sel:
                xy = e['obs'][:, EEF][:, :2]
                ax.plot(xy[:, 0], xy[:, 1], '-', lw=0.7, alpha=0.55, color='tab:green' if e['success'] else 'tab:red')
                ax.plot(*xy[-1], 'o', ms=3, color='tab:green' if e['success'] else 'tab:red')
                ax.plot(*e['obs'][0, NUT][:2], 's', ms=2.5, color='tab:blue', alpha=0.6)
            rate = np.mean([e['success'] for e in sel])
            ax.set_title(f"{titles[name]}\ngoal {'A' if peg == 'left' else 'B'}: {rate:.0%} of {len(sel)} succeed",
                         fontsize=8)
    fig.suptitle('End-effector paths, top-down (green = success, red = failure; blue squares = nut starts; dotted box = '
                 'nut start region).\nThe gripper holds the nut by its handle, 7 cm from the nut center, so paths end '
                 '7 cm past (above) the peg the nut goes on.', fontsize=10)
    fig.tight_layout()
    fig.subplots_adjust(top=0.84, hspace=0.38)
    fig.savefig(f'{out_dir}/trajectories.png', dpi=120)
    print(f'saved {out_dir}/trajectories.png')

    # ---------------- states the base visits
    W = {peg: windows(roll['base'], peg, max_states) for peg in ('left', 'right')}
    _, base, _ = load_checkpoint(f'{R}/m0/base_s0/base/final.ckpt', device)
    base.eval()
    base_plan = {peg: planned(base, W[peg][0], device) for peg in W}
    change, kls = {}, {}
    for label, ckpt in CKPTS.items():
        if not os.path.exists(ckpt):
            print(f'missing {ckpt}'); continue
        _, pol, _ = load_checkpoint(ckpt, device)
        pol.eval()
        for peg in W:
            d = planned(pol, W[peg][0], device) - base_plan[peg]          # (N, 8, 3)
            change[(label, peg)] = d
            kls[(label, peg)] = kl_per_step(pol, base, W[peg][0], device)
        print(f'{label}: change A {np.linalg.norm(change[(label, "left")], axis=-1).mean() * 100:.2f} cm, '
              f'B {np.linalg.norm(change[(label, "right")], axis=-1).mean() * 100:.2f} cm', flush=True)

    # ---------------- figure 2
    fig = plt.figure(figsize=(19, 11))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.85], wspace=0.45, hspace=0.25)
    edges_x, edges_y = np.linspace(-0.28, 0.33, 25), np.linspace(-0.27, 0.27, 22)
    norm = mcolors.LogNorm(vmin=0.03, vmax=3.0)
    for j, label in enumerate(MAP_CKPTS):
        for i, peg in enumerate(('left', 'right')):
            ax = fig.add_subplot(gs[0, 2 * j + i])
            draw_scene(ax)
            eef = W[peg][1]
            d = change[(label, peg)]
            mag = np.linalg.norm(d, axis=-1).mean(1) * 100                  # cm
            vec = d.mean(1)[:, :2] * 100
            ix, iy = np.digitize(eef[:, 0], edges_x) - 1, np.digitize(eef[:, 1], edges_y) - 1
            grid = np.full((len(edges_y) - 1, len(edges_x) - 1), np.nan)
            u, v = np.zeros_like(grid), np.zeros_like(grid)
            for a in range(len(edges_x) - 1):
                for b in range(len(edges_y) - 1):
                    m = (ix == a) & (iy == b)
                    if m.sum() >= 5:
                        grid[b, a] = mag[m].mean(); u[b, a], v[b, a] = vec[m].mean(0)
            im = ax.pcolormesh(edges_x, edges_y, grid, norm=norm, cmap='magma', alpha=0.9, shading='flat')
            cx, cy = (edges_x[:-1] + edges_x[1:]) / 2, (edges_y[:-1] + edges_y[1:]) / 2
            ax.quiver(*np.meshgrid(cx, cy), u, v, color='cyan', scale=8, width=0.004, alpha=0.9)
            ax.set_title(f"{MAP_TITLES[label]}\non {'A' if peg == 'left' else 'B'} states, goal "
                         f"{'A' if peg == 'left' else 'B'}: mean {mag.mean():.2f} cm", fontsize=8)
    cax = fig.add_axes([0.92, 0.56, 0.008, 0.32])
    fig.colorbar(im, cax=cax, label='change of planned positions vs base (cm)')

    # phases
    ax = fig.add_subplot(gs[1, 0:2])
    labels = [l for l in CKPTS if (l, 'left') in change]
    M = np.full((len(labels), 8), np.nan)
    for r_, label in enumerate(labels):
        for c, peg in enumerate(('left', 'right')):
            mag = np.linalg.norm(change[(label, peg)], axis=-1).mean(1) * 100
            for p in range(4):
                m = W[peg][2] == p
                if m.sum() >= 20:
                    M[r_, 4 * c + p] = mag[m].mean()
    im2 = ax.imshow(M, norm=mcolors.LogNorm(vmin=0.03, vmax=max(3.0, np.nanmax(M))), cmap='magma', aspect='auto')
    for r_ in range(M.shape[0]):
        for c in range(8):
            if not np.isnan(M[r_, c]):
                ax.text(c, r_, f'{M[r_, c]:.2f}', ha='center', va='center', fontsize=7,
                        color='white' if M[r_, c] < 0.5 else 'black')
    ax.set_xticks(range(8)); ax.set_xticklabels([f'A: {p}' for p in PHASES] + [f'B: {p}' for p in PHASES],
                                                rotation=30, fontsize=8)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(3.5, color='w', lw=2)
    ax.set_title('Mean change of planned positions (cm) by task phase, on the base\'s own states', fontsize=9)
    fig.colorbar(im2, ax=ax, fraction=0.03)

    # noise levels
    ax = fig.add_subplot(gs[1, 2:4])
    K = len(next(iter(kls.values())))
    N_ = np.array([[*kls[(l, 'left')], *kls[(l, 'right')]] for l in labels])
    im3 = ax.imshow(N_, norm=mcolors.LogNorm(vmin=max(1e-3, np.nanmin(N_[N_ > 0])), vmax=np.nanmax(N_)),
                    cmap='viridis', aspect='auto')
    ax.set_xticks(range(2 * K)); ax.set_xticklabels([f'A{k}' for k in range(K)] + [f'B{k}' for k in range(K)],
                                                    fontsize=7)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels([])
    ax.axvline(K - 0.5, color='w', lw=2)
    ax.set_title('KL from the base per denoising step (0 = pure noise ... 9 = final), A vs B states\n'
                 '(same rows as left)', fontsize=9)
    fig.colorbar(im3, ax=ax, fraction=0.03)
    fig.suptitle("Where post-training on A changes the policy's plans (seed 0; states from the base's own rollouts).\n"
                 "SFT = supervised fine-tuning on expert A demos at lr 1e-6; checkpoints by gradient steps. PPO and SFT have no "
                 "common step unit, so maps compare them at equal KL from the base.", fontsize=10)
    fig.savefig(f'{out_dir}/action_changes.png', dpi=120, bbox_inches='tight')
    print(f'saved {out_dir}/action_changes.png')
    np.savez(f'{out_dir}/phase_table.npz', M=M, labels=np.array(labels), N=N_)


if __name__ == '__main__':
    main()
