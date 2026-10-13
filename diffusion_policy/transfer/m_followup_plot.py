"""
Summary figure for the three follow-up experiments (mean ± SE over seeds).

  (a) first pair: PPO vs BC held to PPO's KL from the base - change on A and on B
  (b) first pair: PPO on B from each post-trained checkpoint - B success along B training
  (c) second pair (B = round nut, ring grasp): change on B vs change on A along training

  python -m transfer.m_followup_plot [--out results/followups.png]
"""

import json
import os

import click
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

R = '/iris/u/lucyshi/fpl_transfer/runs'
SEEDS = (0, 1, 2)


def res(p):
    return json.load(open(p))['results'] if os.path.exists(p) else None


def ms(v):
    v = [x for x in v if x is not None]
    if not v:
        return np.nan, 0.0
    return float(np.mean(v)), (float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0)


def delta(path, base, peg):
    out = []
    for s in SEEDS:
        r, b = res(path(s)), res(base(s))
        out.append(r[peg]['success'] - b[peg]['success'] if r and b else None)
    return ms(out)


@click.command()
@click.option('--out', default='/iris/u/lucyshi/fpl_transfer/results/followups.png')
def main(out):
    mb = lambda s: f'{R}/m0/base_s{s}/eval.json'
    xb = lambda s: f'{R}/x0/base_s{s}/eval.json'
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 5.8))

    # (a) KL-held BC
    arms = [('PPO\n3000 ep.', lambda s: f'{R}/m2/s{s}/ppo/n3000_eval.json', 'tab:red'),
            ('filtered BC\n≤ PPO-3000 KL', lambda s: f'{R}/m7/s{s}/filtered_bc_klppo3000/step5000_eval.json', 'tab:orange'),
            ('SFT expert\n≤ PPO-3000 KL', lambda s: f'{R}/m7/s{s}/sft_expert_klppo3000/step5000_eval.json', 'tab:gray'),
            ('PPO\n6000 ep.', lambda s: f'{R}/m2/s{s}/ppo/n6000_eval.json', 'tab:red'),
            ('filtered BC\n≤ PPO-6000 KL', lambda s: f'{R}/m7/s{s}/filtered_bc_klppo6000/step5000_eval.json', 'tab:orange'),
            ('SFT expert\n≤ PPO-6000 KL', lambda s: f'{R}/m7/s{s}/sft_expert_klppo6000/step5000_eval.json', 'tab:gray')]
    x = np.arange(len(arms))
    for i, (name, path, c) in enumerate(arms):
        (a, ae), (b, be) = delta(path, mb, 'left'), delta(path, mb, 'right')
        ax1.bar(i - 0.18, a, 0.34, yerr=ae, color=c, alpha=0.45, capsize=3, label='Δ on A' if i == 0 else None,
                edgecolor='k', hatch='//')
        ax1.bar(i + 0.18, b, 0.34, yerr=be, color=c, alpha=0.95, capsize=3, label='Δ on B' if i == 0 else None,
                edgecolor='k')
    ax1.axhline(0, color='k', lw=0.8)
    ax1.axvline(2.5, color='gray', lw=0.8, ls=':')
    ax1.set_xticks(x); ax1.set_xticklabels([a[0] for a in arms], fontsize=7.5)
    ax1.set_ylabel('change in success vs base')
    ax1.set_title("(a) BC held to PPO's KL from the base (no replay)\nhatched = A (post-trained), solid = B", fontsize=10)
    ax1.legend(fontsize=8, loc='lower left')

    # (b) PPO on B
    starts = [('base', 'base', mb, 'k'), ('ppoA6000', 'PPO on A (6000 ep.)', lambda s: f'{R}/m2/s{s}/ppo/n6000_eval.json', 'tab:red'),
              ('sft_expert_r1000', 'SFT expert N=1000 (replay)', lambda s: f'{R}/m1r/s{s}/sft_expert/n1000_eval.json', 'tab:gray'),
              ('filtered_bc_r1000', 'filtered BC N=1000 (replay)', lambda s: f'{R}/m1r/s{s}/filtered_bc/n1000_eval.json', 'tab:orange'),
              ('awr_r1000', 'AWR N=1000 (replay)', lambda s: f'{R}/m1r/s{s}/awr/n1000_eval.json', 'tab:green')]
    ns = (0, 200, 1000, 3000)
    for key, label, start, c in starts:
        pts = [ms([(res(start(s)) or {}).get('right', {}).get('success') for s in SEEDS])]
        pts += [ms([(res(f'{R}/m6/s{s}/{key}/n{n}_eval.json') or {}).get('right', {}).get('success') for s in SEEDS])
                for n in ns[1:]]
        ax2.errorbar(ns, [p[0] for p in pts], yerr=[p[1] for p in pts], fmt='-o', color=c, capsize=3, label=label)
    ax2.set_xlabel('PPO episodes on B'); ax2.set_ylabel('B success')
    ax2.set_ylim(0.5, 1.0)
    ax2.set_title('(b) PPO on B, starting from checkpoints post-trained on A', fontsize=10)
    ax2.legend(fontsize=8, loc='lower right')

    # (c) second pair
    def path_pts(ax, items, color, marker, label, ls='-', mfc=None):
        pts = [(0.0, 0.0, 0.0, 0.0)]
        for p in items:
            (a, ae), (b, be) = delta(p, xb, 'left'), delta(p, xb, 'round')
            pts.append((a, b, ae, be))
        ax.plot([p[0] for p in pts], [p[1] for p in pts], ls, color=color, alpha=0.6)
        for i, (a, b, ae, be) in enumerate(pts[1:]):
            ax.errorbar(a, b, xerr=ae, yerr=be, fmt=marker, color=color, mfc=mfc or color, capsize=2,
                        label=label if i == 0 else None)
    path_pts(ax3, [lambda s, n=n: f'{R}/x2/s{s}/ppo/n{n}_eval.json' for n in (1000, 3000, 6000)], 'tab:red', '^',
             'PPO (1000 / 3000 / 6000 ep.)')
    for m, c in (('sft_expert', 'tab:gray'), ('filtered_bc', 'tab:orange')):
        path_pts(ax3, [lambda s, st=st, m=m: f'{R}/x5/s{s}/{m}_lr1e-06early/step{st}_eval.json' for st in (200, 400, 700, 1000)],
                 c, 'o', f'{m.replace("_", " ")}, lr 1e-6, no replay', ls='--', mfc='white')
        (a, ae), (b, be) = delta(lambda s, m=m: f'{R}/x1r/s{s}/{m}/n1000_eval.json', xb, 'left'), \
            delta(lambda s, m=m: f'{R}/x1r/s{s}/{m}/n1000_eval.json', xb, 'round')
        ax3.errorbar(a, b, xerr=ae, yerr=be, fmt='s', color=c, ms=8, capsize=2, label=f'{m.replace("_", " ")} N=1000, replay')
    ax3.axhline(0, color='k', lw=0.6); ax3.axvline(0, color='k', lw=0.6)
    (x0, x1), (y0, y1) = ax3.get_xlim(), ax3.get_ylim()
    ax3.plot([min(x0, y0), max(x1, y1)], [min(x0, y0), max(x1, y1)], color='k', lw=0.6, alpha=0.4)
    ax3.set_xlim(x0, x1); ax3.set_ylim(y0, y1)
    ax3.set_xlabel('Δ success on A (square nut, post-trained)'); ax3.set_ylabel('Δ success on B (round nut, ring grasp)')
    ax3.set_title('(c) Second task pair: B = round nut grasped by its ring\n(grey line: ΔB = ΔA)', fontsize=10)
    ax3.legend(fontsize=7.5, loc='lower left')
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
