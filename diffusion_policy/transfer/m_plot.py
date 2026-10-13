"""
Figures for the redesigned transfer experiment (mean +/- SE over seeds; changes are
against each seed's base policy on the same start states).

  (a) no replay: change on task B against change on task A along training - PPO vs
      supervised post-training (offline at three learning rates, and iterated on the
      policy's own rollouts)
  (b) with replay of the pretraining pool: offline arms at N = 200 / 1000
  (c) task B success after fine-tuning on K clean B demos, against K

  python -m transfer.m_plot [--seeds 0,1,2] [--out results/m_transfer.png]
"""

import json
import os

import click
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

R = '/iris/u/lucyshi/fpl_transfer/runs'
KS = (3, 5, 10)
LEARN_B_K = 10
COLORS = {'sft_pool': 'tab:blue', 'filtered_bc': 'tab:orange', 'awr': 'tab:green', 'udrl_scalar': 'tab:purple',
          'udrl_axes': 'tab:pink', 'sft_expert': 'tab:gray', 'ppo': 'tab:red'}
LABELS = {'sft_pool': 'SFT (pool)', 'filtered_bc': 'filtered BC', 'awr': 'AWR', 'udrl_scalar': 'UDRL (scalar)',
          'udrl_axes': 'UDRL (axes)', 'sft_expert': 'SFT (expert demos)', 'ppo': 'PPO'}
LR_STYLE = {'1e-04': ':', '1e-05': '-.', '1e-06early': '--', '1e-06': '--'}
M5 = {'1e-04': (100, 200, 500, 1000), '1e-05': (500, 1000, 2500, 5000),
      '1e-06early': (100, 200, 400, 700, 1000), '1e-06': (1000, 3000, 6000, 10000)}


def res(p):
    return json.load(open(p))['results'] if os.path.exists(p) else None


def succ(r, peg):
    return r[peg]['success'] if r and r.get(peg) else None


def mean_se(v):
    v = [x for x in v if x is not None]
    if not v:
        return None, None
    return float(np.mean(v)), (float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0)


def learn(job_dir, m3_dir, tag, k):
    name = f'{tag}_learnB' + ('' if k == LEARN_B_K else f'_k{k}') + '_eval.json'
    return res(f'{job_dir}/{name}') or res(f'{m3_dir}/{name}')


@click.command()
@click.option('--seeds', default='0,1,2')
@click.option('--out', default='/iris/u/lucyshi/fpl_transfer/results/m_transfer.png')
@click.option('--min_seeds', default=2, help='Plot a point only when this many seeds have it')
def main(seeds, out, min_seeds):
    seeds = [int(s) for s in seeds.split(',')]
    base = {s: res(f'{R}/m0/base_s{s}/eval.json') for s in seeds}

    def delta(path_fn):
        """mean/SE over seeds of (dA, dB) for an eval path function of the seed."""
        ev = {s: res(path_fn(s)) for s in seeds}
        ok = [s for s in seeds if ev[s] and base[s]]
        if len(ok) < min_seeds:
            return None
        return (mean_se([succ(ev[s], 'left') - succ(base[s], 'left') for s in ok]),
                mean_se([succ(ev[s], 'right') - succ(base[s], 'right') for s in ok]))

    def path(ax, pts, color, ls, marker, label, mfc=None, annotate=None):
        pts = [p for p in pts if p[1] is not None]
        if not pts:
            return
        xs = [0] + [p[1][0][0] for p in pts]
        ys = [0] + [p[1][1][0] for p in pts]
        ax.plot(xs, ys, ls, color=color, alpha=0.6, lw=1.5)
        for i, (n, (dA, dB)) in enumerate(pts):
            ax.errorbar(dA[0], dB[0], xerr=dA[1], yerr=dB[1], fmt=marker, ms=6, color=color,
                        mfc=mfc or color, capsize=2, label=label if i == len(pts) - 1 else None)
            if annotate:
                ax.annotate(annotate(n), (dA[0], dB[0]), textcoords='offset points', xytext=(4, 3), fontsize=7,
                            color=color)

    def diagonal(ax):
        ax.axhline(0, color='k', lw=0.6); ax.axvline(0, color='k', lw=0.6)
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        lo, hi = min(x0, y0), max(x1, y1)
        ax.plot([lo, hi], [lo, hi], color='k', lw=0.6, alpha=0.4)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_xlabel('Δ success on A (post-trained task)'); ax.set_ylabel('Δ success on B (zero-shot)')

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6.2))

    # ---- (a) no replay: paths along training
    ppo = [(n, delta(lambda s, n=n: f'{R}/m2/s{s}/ppo/n{n}_eval.json')) for n in (200, 1000, 3000, 6000)]
    path(ax1, ppo, COLORS['ppo'], '-', '^', 'PPO (lr 1e-6), episodes', annotate=str)
    for m in ('filtered_bc', 'sft_expert'):
        for lr in ('1e-04', '1e-05', '1e-06early'):
            pts = [(st, delta(lambda s, st=st: f'{R}/m5/s{s}/{m}_lr{lr}/step{st}_eval.json')) for st in M5[lr]]
            if lr == '1e-06early':     # continue the early-steps run's path with the longer run at the same lr
                pts += [(st, delta(lambda s, st=st: f'{R}/m5/s{s}/{m}_lr1e-06/step{st}_eval.json'))
                        for st in M5['1e-06'][1:]]
            path(ax1, pts, COLORS[m], LR_STYLE[lr], 'o', f"{LABELS[m]}, offline, lr {lr.replace('early', '')}",
                 mfc='white')
    for m, tag, ns, ls in (('filtered_bc', '', (200, 1000, 3000), ':'),
                           ('filtered_bc', '_lr1e-06', (200, 400, 600, 800, 1000), '--'),
                           ('awr', '', (200, 1000, 3000), ':')):
        pts = [(n, delta(lambda s, n=n: f'{R}/m4/s{s}/{m}{tag}/n{n}_eval.json')) for n in ns]
        lr = '1e-6' if tag else '1e-4'
        path(ax1, pts, COLORS[m], ls, 'D', f'{LABELS[m]}, iterated on own rollouts, lr {lr}', annotate=str)
    diagonal(ax1)
    ax1.set_title('(a) No replay: B vs A along training\n(grey line: ΔB = ΔA; labels: episodes)')
    ax1.legend(fontsize=7, loc='lower left')

    # ---- (b) with replay: offline arms
    for m in COLORS:
        if m == 'ppo':
            continue
        pts = [(n, delta(lambda s, n=n: f'{R}/m1r/s{s}/{m}/n{n}_eval.json')) for n in (200, 1000)]
        path(ax2, pts, COLORS[m], '-', 'o', LABELS[m], annotate=lambda n: f'N={n}')
    path(ax2, ppo, COLORS['ppo'], '-', '^', 'PPO (no replay), for reference', mfc='white', annotate=str)
    diagonal(ax2)
    ax2.set_title('(b) Offline on the shared A pool, with 50% replay of pretraining data\n(N = A-pool episodes)')
    ax2.legend(fontsize=7, loc='upper left')

    # ---- (c) learnability
    bl = [mean_se([succ(learn(f'{R}/m0/base_s{s}', f'{R}/m3/m0/s{s}/base', 'base', k), 'right') for s in seeds])
          for k in KS]
    ax3.errorbar(KS, [b[0] if b[0] is not None else np.nan for b in bl], yerr=[b[1] or 0 for b in bl], fmt='-o',
                 color='k', lw=2, capsize=2, label='base policy')
    curves = [('m2', 'ppo', 'n6000', R + '/m2/s{s}/ppo', R + '/m3/m2/s{s}/ppo', 'PPO, 6000 ep. (no replay)', '-^')]
    curves += [('m1r', m, 'n1000', R + '/m1r/s{s}/' + m, R + '/m3/m1r/s{s}/' + m, f'{LABELS[m]}, N=1000 (replay)',
                '-o') for m in ('filtered_bc', 'awr', 'sft_expert', 'sft_pool')]
    curves += [('m1', m, 'n1000', R + '/m1/s{s}/' + m, R + '/m1/s{s}/' + m, f'{LABELS[m]}, N=1000 (no replay)',
                '--o') for m in ('filtered_bc', 'sft_expert')]
    for stage, m, tag, jd, m3d, label, fmt in curves:
        pts = []
        for k in KS:
            v = [succ(learn(jd.format(s=s), m3d.format(s=s), tag, k), 'right') for s in seeds]
            pts.append(mean_se(v) if sum(x is not None for x in v) >= min_seeds else (None, None))
        if all(p[0] is None for p in pts):
            continue
        ax3.errorbar(KS, [p[0] if p[0] is not None else np.nan for p in pts], yerr=[p[1] or 0 for p in pts],
                     fmt=fmt, color=COLORS[m], capsize=2, alpha=0.85, label=label,
                     mfc='white' if stage == 'm1' else COLORS[m])
    ax3.set_xscale('log'); ax3.set_xticks(KS); ax3.set_xticklabels([str(k) for k in KS])
    ax3.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax3.set_ylim(-0.02, 1.02)
    ax3.set_xlabel('clean B demos for fine-tuning (K)'); ax3.set_ylabel('B success after fine-tuning')
    ax3.set_title('(c) Learning B after post-training on A')
    ax3.legend(fontsize=7, loc='upper left')

    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
