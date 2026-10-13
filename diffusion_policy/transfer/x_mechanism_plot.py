"""
Why PPO on A hurts the round-nut task and SFT does not (second pair; recorded rollouts in runs/xrec).

  (a) round-nut and square-nut success against how far the nut center is from the peg center at release
  (b) release offsets on A and B: base, PPO, SFT medians - PPO moves both the same way
  (c) outcomes on B: success / misplaced at the peg / missed grasp / dropped

  python -m transfer.x_mechanism_plot [--out results/geometry/mechanism.png]
"""

import click
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

R = '/iris/u/lucyshi/fpl_transfer/runs/xrec'
PEG = {'left': np.array([0.23, 0.10]), 'round': np.array([0.23, -0.10])}
ARMS = {'base': ('base', 'k'), 'ppo6000': ('PPO, 6000 ep.', 'tab:red'), 'sft_replay': ('SFT expert, replay', 'tab:gray'),
        'sft_lr1e6_200': ('SFT expert, lr 1e-6, 200 steps', 'tab:blue')}


def load(arm, task):
    out = []
    for s in (0, 1, 2):
        try:
            f = h5py.File(f'{R}/{arm}_s{s}/rollouts.hdf5', 'r')
        except OSError:
            continue
        with f:
            for k in f['data']:
                d = f['data'][k]
                peg = d.attrs['target_peg']
                peg = peg if isinstance(peg, str) else peg.decode()
                if peg != task:
                    continue
                o, a, ok = d['obs']['object'][:], d['actions'][:], bool(d.attrs['success'])
                up = np.nonzero(o[:, 2] > o[0, 2] + 0.03)[0]
                if not len(up):
                    out.append({'outcome': 'missed grasp', 'ok': ok}); continue
                opens = np.nonzero(a[up[0]:, 6] < 0)[0]
                t = up[0] + opens[0] if len(opens) else len(a) - 1
                off = (o[t, :2] - PEG[task]) * 100
                outcome = 'success' if ok else ('misplaced at the peg' if np.abs(off).max() < 3 else 'dropped / away')
                out.append({'outcome': outcome, 'ok': ok, 'off': off})
    return out


@click.command()
@click.option('--out', default='/iris/u/lucyshi/fpl_transfer/results/geometry/mechanism.png')
def main(out):
    data = {(arm, t): load(arm, t) for arm in ARMS for t in ('left', 'round')}
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.3), gridspec_kw={'width_ratios': [1, 1, 1, 1.1]})

    # (a) success vs release offset
    ax = axes[0]
    bins = np.array([0, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0])
    for task, c, name in (('round', 'tab:purple', 'B: round nut on round peg'), ('left', 'tab:olive', 'A: square nut on square peg')):
        pts = [e for arm in ARMS for e in data[(arm, task)] if 'off' in e and np.abs(e['off']).max() < 3]
        r = np.array([np.linalg.norm(e['off']) for e in pts]); ok = np.array([e['ok'] for e in pts])
        mids, rates = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (r >= lo) & (r < hi)
            if m.sum() >= 10:
                mids.append((lo + hi) / 2); rates.append(ok[m].mean())
        ax.plot(mids, rates, '-o', color=c, label=name)
    for arm in ('base', 'ppo6000'):
        offs = np.array([e['off'] for e in data[(arm, 'round')] if 'off' in e and np.abs(e['off']).max() < 3])
        med = np.linalg.norm(np.median(offs, axis=0))
        ax.axvline(med, color=ARMS[arm][1], ls='--', lw=1.2)
        ax.text(med + 0.03, 0.05 if arm == 'base' else 0.15, f'{ARMS[arm][0]}\nround-nut median', color=ARMS[arm][1], fontsize=7.5)
    ax.set_xlabel('nut center to peg center at release (cm)'); ax.set_ylabel('success')
    ax.set_title('(a) How precise the release must be\n(round nut: sharp edge at ~1.5 cm)', fontsize=10)
    ax.legend(fontsize=8, loc='lower left')

    # (b) release offsets with medians, A and B
    for j, (task, title) in enumerate((('left', 'A: square nut'), ('round', 'B: round nut'))):
        ax = axes[1 + j]
        for arm in ('base', 'ppo6000', 'sft_replay'):
            name, c = ARMS[arm]
            offs = np.array([e['off'] for e in data[(arm, task)] if 'off' in e and np.abs(e['off']).max() < 3])
            oks = np.array([e['ok'] for e in data[(arm, task)] if 'off' in e and np.abs(e['off']).max() < 3])
            ax.scatter(offs[oks, 0], offs[oks, 1], s=7, color=c, alpha=0.25)
            ax.scatter(offs[~oks, 0], offs[~oks, 1], s=14, color=c, alpha=0.6, marker='x')
            med = np.median(offs, axis=0)
            ax.plot(*med, 'o', ms=11, color=c, mec='w', mew=1.5, label=f'{name}: median ({med[0]:+.2f}, {med[1]:+.2f})', zorder=5)
        mb = np.median([e['off'] for e in data[('base', task)] if 'off' in e and np.abs(e['off']).max() < 3], axis=0)
        mp = np.median([e['off'] for e in data[('ppo6000', task)] if 'off' in e and np.abs(e['off']).max() < 3], axis=0)
        ax.annotate('', xy=mp, xytext=mb, arrowprops=dict(arrowstyle='->', color='tab:red', lw=2), zorder=6)
        if task == 'round':
            ax.add_patch(plt.Circle((0, 0), 1.5, fill=False, color='tab:purple', ls='--'))
            ax.text(-1.45, 1.6, 'success edge ~1.5 cm', color='tab:purple', fontsize=8)
        ax.plot(0, 0, '+', color='k', ms=14)
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect('equal')
        ax.set_xlabel('nut center − peg center, x (cm)'); ax.set_ylabel('y (cm)')
        ax.set_title(f'(b) Release positions, {title}\n(dots success, x fail; red arrow: base → PPO median)', fontsize=10)
        ax.legend(fontsize=7, loc='lower left')

    # (c) outcomes on B
    ax = axes[3]
    outcomes = [('success', 'tab:green'), ('misplaced at the peg', 'tab:orange'), ('missed grasp', 'tab:red'), ('dropped / away', 'tab:brown')]
    names = list(ARMS)
    bottom = np.zeros(len(names))
    for o, c in outcomes:
        vals = np.array([np.mean([e['outcome'] == o for e in data[(a, 'round')]]) for a in names])
        ax.bar(range(len(names)), vals, bottom=bottom, color=c, label=o, edgecolor='w')
        for i, v in enumerate(vals):
            if v > 0.06:
                ax.text(i, bottom[i] + v / 2, f'{v:.2f}', ha='center', va='center', fontsize=8, color='w')
        bottom += vals
    ax.set_xticks(range(len(names))); ax.set_xticklabels([ARMS[a][0].replace(', ', ',\n') for a in names], fontsize=8)
    ax.set_ylabel('fraction of episodes'); ax.set_ylim(0, 1)
    ax.set_title('(c) What happens on the round-nut task\n(300 episodes per arm, 3 seeds)', fontsize=10)
    ax.legend(fontsize=7.5, loc='upper right', bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
