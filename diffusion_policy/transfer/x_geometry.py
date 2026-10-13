"""
Grasp and release geometry of recorded rollouts (transfer.experiments record), second task pair.

Per episode:
  grasp   - first step the gripper is commanded to close: gripper position relative to the nut's center
            in the nut's frame (x along the handle), its height, and whether the nut is lifted afterwards
  release - first step the gripper is commanded to open after a lift: the nut center's xy offset from the
            target peg, its height above the table and its tilt
  outcome - missed grasp (closed, never lifted) / dropped (lifted, released or fallen more than 5 cm from
            the peg) / misplaced (released near the peg, not on it) / success

  python -m transfer.x_geometry [--arms base,ppo6000,sft_replay,sft_lr1e6_200] [--out_dir results/geometry]
"""

import json
import os

import click
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

R = '/iris/u/lucyshi/fpl_transfer/runs'
PEG_XY = {'left': np.array([0.23, 0.10]), 'round': np.array([0.23, -0.10])}
TABLE_Z = 0.8
NUT_POS, NUT_QUAT, EEF_POS, EEF_QUAT = slice(0, 3), slice(3, 7), slice(14, 17), slice(17, 21)
LABELS = {'base': 'base', 'ppo6000': 'PPO, 6000 ep.', 'sft_replay': 'SFT expert, replay',
          'sft_lr1e6_200': 'SFT expert, lr 1e-6, 200 steps'}
COLORS = {'base': 'k', 'ppo6000': 'tab:red', 'sft_replay': 'tab:gray', 'sft_lr1e6_200': 'tab:blue'}


def yaw_of(q):
    """Yaw of a robosuite (x, y, z, w) quaternion's x axis."""
    x, y, z, w = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def tilt_of(q):
    """Angle (deg) between the body's z axis and the world z axis."""
    x, y, z, w = q
    return np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))


def episode_geometry(obs, act, peg):
    nut, eef = obs[:, NUT_POS], obs[:, EEF_POS]
    z0 = nut[0, 2]
    out = {}
    closes = np.nonzero(act[:, 6] > 0)[0]
    if not len(closes):
        return {'outcome': 'never closed'}
    tg = closes[0]
    yaw = yaw_of(obs[tg, NUT_QUAT])
    c, s = np.cos(-yaw), np.sin(-yaw)
    rel = eef[tg] - nut[tg]
    out['grasp_nut_frame'] = np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1], rel[2]])
    out['grasp_world'] = rel
    lifted = np.nonzero(nut[tg:, 2] > z0 + 0.03)[0]
    if not len(lifted):
        out['outcome'] = 'missed grasp'
        return out
    tl = tg + lifted[0]
    opens = np.nonzero(act[tl:, 6] < 0)[0]
    tr = tl + opens[0] if len(opens) else len(act) - 1
    err = nut[tr, :2] - PEG_XY[peg]
    out.update(release_err=err, release_height=nut[tr, 2] - TABLE_Z, release_tilt=tilt_of(obs[tr, NUT_QUAT]))
    # transport: eef speed while carrying, grasp-to-release time, and how far the nut moved in the hand
    carry = slice(tl, max(tl + 1, tr))
    out['carry_speed'] = float(np.linalg.norm(np.diff(eef[carry], axis=0), axis=1).mean()) if tr > tl + 1 else np.nan
    out['carry_steps'] = int(tr - tg)
    d_hand = np.linalg.norm(nut[carry] - eef[carry], axis=1)
    out['slip'] = float(d_hand.max() - d_hand[0]) if len(d_hand) else np.nan
    if np.linalg.norm(err) > 0.05:
        out['outcome'] = 'dropped / released away from peg'
    return out


def load(arm, seeds):
    eps = {'left': [], 'round': []}
    for s in seeds:
        path = f'{R}/xrec/{arm}_s{s}/rollouts.hdf5'
        if not os.path.exists(path):
            continue
        with h5py.File(path, 'r') as f:
            for k in f['data']:
                d = f['data'][k]
                peg = d.attrs['target_peg']
                peg = peg if isinstance(peg, str) else peg.decode()
                obs = np.concatenate([d['obs'][key][:] for key in ('object', 'robot0_eef_pos', 'robot0_eef_quat',
                                                                  'robot0_gripper_qpos')], axis=-1)
                g = episode_geometry(obs, d['actions'][:], peg)
                g['success'] = bool(d.attrs['success'])
                if 'outcome' not in g:
                    g['outcome'] = 'success' if g['success'] else 'misplaced at the peg'
                eps[peg].append(g)
    return eps


@click.command()
@click.option('--arms', default='base,ppo6000,sft_replay,sft_lr1e6_200')
@click.option('--seeds', default='0,1,2')
@click.option('--out_dir', default='/iris/u/lucyshi/fpl_transfer/results/geometry')
def main(arms, seeds, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    seeds = [int(s) for s in seeds.split(',')]
    data = {a: load(a, seeds) for a in arms.split(',')}
    outcomes = ('success', 'misplaced at the peg', 'dropped / released away from peg', 'missed grasp', 'never closed')
    summary = {}
    for task in ('left', 'round'):
        print(f"\n### {'A (square nut, handle grasp)' if task == 'left' else 'B (round nut, ring grasp)'}\n")
        print('| arm | n | ' + ' | '.join(outcomes) + ' | grasp x, y, z in nut frame (cm) | release error x, y (cm) | '
              'release height (cm) | release tilt (deg) | carry speed (mm/step) | grasp-to-release steps | '
              'in-hand slip (cm) |')
        print('|---|---|' + '---|' * (len(outcomes) + 7))
        for a, eps in data.items():
            e = eps[task]
            if not e:
                continue
            frac = [np.mean([x['outcome'] == o for x in e]) for o in outcomes]
            g = np.array([x['grasp_nut_frame'] for x in e if 'grasp_nut_frame' in x]) * 100
            r = [x for x in e if 'release_err' in x]
            re = np.array([x['release_err'] for x in r]) * 100
            rh = np.array([x['release_height'] for x in r]) * 100
            rt = np.array([x['release_tilt'] for x in r])
            summary[(a, task)] = {'grasp': g, 'release': re, 'eps': e}
            print(f"| {LABELS[a]} | {len(e)} | " + ' | '.join(f'{f:.2f}' for f in frac)
                  + f" | {g[:, 0].mean():+.1f}, {g[:, 1].mean():+.1f}, {g[:, 2].mean():+.1f} (sd {g[:, 0].std():.1f}, {g[:, 1].std():.1f}, {g[:, 2].std():.1f})"
                  + f" | {re[:, 0].mean():+.1f}, {re[:, 1].mean():+.1f} (sd {re[:, 0].std():.1f}, {re[:, 1].std():.1f})"
                  + f" | {rh.mean():.1f} | {rt.mean():.0f} | {np.nanmean([x['carry_speed'] for x in r]) * 1000:.1f}"
                  + f" | {np.mean([x['carry_steps'] for x in r]):.0f} | {np.nanmean([x['slip'] for x in r]) * 100:.1f} |")

    # shifts vs base, A and B, in the world frame
    print('\n### Mean shift vs base (world frame, cm): grasp (gripper relative to nut) and release (nut relative to peg)\n')
    print('| arm | A grasp shift | B grasp shift | A release shift | B release shift |')
    print('|---|---|---|---|---|')
    for a in data:
        if a == 'base' or (a, 'left') not in summary:
            continue
        row = []
        for kind in ('grasp', 'release'):
            for task in ('left', 'round'):
                if kind == 'grasp':
                    cur = np.array([x['grasp_world'] for x in summary[(a, task)]['eps'] if 'grasp_world' in x])[:, :2] * 100
                    ref = np.array([x['grasp_world'] for x in summary[('base', task)]['eps'] if 'grasp_world' in x])[:, :2] * 100
                else:
                    cur, ref = summary[(a, task)]['release'], summary[('base', task)]['release']
                d = cur.mean(0) - ref.mean(0)
                row.append(f'({d[0]:+.1f}, {d[1]:+.1f})')
        print(f'| {LABELS[a]} | {row[0]} | {row[1]} | {row[2]} | {row[3]} |')

    # figure: release positions relative to the peg, and grasp positions in the nut frame
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    for j, task in enumerate(('left', 'round')):
        ax = axes[0, j]
        for a in data:
            if (a, task) not in summary:
                continue
            e = summary[(a, task)]['eps']
            pts = np.array([x['release_err'] for x in e if 'release_err' in x]) * 100
            ok = np.array([x['success'] for x in e if 'release_err' in x])
            ax.scatter(pts[ok, 0], pts[ok, 1], s=14, color=COLORS[a], alpha=0.6, marker='o', label=f'{LABELS[a]} (success)')
            ax.scatter(pts[~ok, 0], pts[~ok, 1], s=22, color=COLORS[a], alpha=0.8, marker='x', label=f'{LABELS[a]} (fail)')
        ax.add_patch(plt.Circle((0, 0), 3.0, fill=False, ls=':', color='k'))
        ax.set_xlim(-8, 8); ax.set_ylim(-8, 8); ax.set_aspect('equal')
        ax.set_xlabel('nut center − peg, x (cm)'); ax.set_ylabel('y (cm)')
        ax.set_title(f"Where the nut is released, task {'A (square)' if task == 'left' else 'B (round)'}\n"
                     f"(dotted: ±3 cm success window)", fontsize=10)
        ax.legend(fontsize=6.5, loc='upper right')
        ax = axes[1, j]
        for a in data:
            if (a, task) not in summary:
                continue
            e = summary[(a, task)]['eps']
            pts = np.array([x['grasp_nut_frame'] for x in e if 'grasp_nut_frame' in x]) * 100
            lifted = np.array([x['outcome'] != 'missed grasp' for x in e if 'grasp_nut_frame' in x])
            ax.scatter(pts[lifted, 0], pts[lifted, 1], s=14, color=COLORS[a], alpha=0.6, marker='o', label=f'{LABELS[a]} (lifted)')
            ax.scatter(pts[~lifted, 0], pts[~lifted, 1], s=22, color=COLORS[a], alpha=0.8, marker='x', label=f'{LABELS[a]} (missed)')
        if task == 'left':
            ax.add_patch(plt.Rectangle((2.875, -1.59), 5.05, 3.18, fill=False, color='k', ls=':'))   # square nut handle
            ax.text(3.0, 2.0, 'handle', fontsize=8)
        else:
            ax.add_patch(plt.Rectangle((-5.37, -2.25), 2.25, 4.5, fill=False, color='k', ls=':'))   # far ring wall
            ax.add_patch(plt.Rectangle((3.475, -1.59), 5.05, 3.18, fill=False, color='gray', ls=':'))  # handle
            ax.text(-5.3, 2.6, 'ring wall', fontsize=8); ax.text(3.6, 2.0, 'handle', fontsize=8, color='gray')
        ax.set_xlim(-9, 11); ax.set_ylim(-6, 6); ax.set_aspect('equal')
        ax.set_xlabel('gripper − nut center along the nut\'s handle axis (cm)'); ax.set_ylabel('across (cm)')
        ax.set_title(f"Where the gripper closes, task {'A (square)' if task == 'left' else 'B (round)'} (nut frame)", fontsize=10)
        ax.legend(fontsize=6.5, loc='lower left')
    fig.tight_layout()
    fig.savefig(f'{out_dir}/geometry.png', dpi=120)
    print(f'\nsaved {out_dir}/geometry.png')


if __name__ == '__main__':
    main()
