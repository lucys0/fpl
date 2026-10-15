"""
Per-task views of the K-task experiments (objects as rows, targets as columns; letters mark the split:
R eligible for RL, H held out from RL, U never in pretraining).

  (top)    base success for each seed
  (middle) mean change after PPO (32k episodes) for K = 1, 4, 16, averaged over runs (cells a run trained on
           are left out of that run's average, so every cell shows transfer only)
  (bottom) the K = 1 transfer matrix: one row per run (its trained task boxed), change on every task

  python -m ktask.plot_tasks [--out results/tasks.png]
"""

import click
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ktask.analyze import base_eval, e1_runs, load_eval
from ktask.data import SPLITS
from ktask.env import N_OBJ, N_TGT, OBJ_NAMES, TGT_NAMES, task_name

SPLIT_OF = {t: s for s, ts in SPLITS.items() for t in ts}


def grid(values):
    g = np.full((N_OBJ, N_TGT), np.nan)
    for t, v in values.items():
        g[t // N_TGT, t % N_TGT] = v
    return g


def draw(ax, g, title, cmap, vmin, vmax, fmt='{:.2f}'):
    im = ax.imshow(g, cmap=cmap, vmin=vmin, vmax=vmax)
    for o in range(N_OBJ):
        for t in range(N_TGT):
            v = g[o, t]
            s = SPLIT_OF[o * N_TGT + t]
            ax.text(t, o, (fmt.format(v) if not np.isnan(v) else '') + f'\n{s}', ha='center', va='center',
                    fontsize=7, color='k')
    ax.set_xticks(range(N_TGT))
    ax.set_xticklabels(TGT_NAMES, fontsize=8)
    ax.set_yticks(range(N_OBJ))
    ax.set_yticklabels(OBJ_NAMES, fontsize=8)
    ax.set_title(title, fontsize=9.5)
    return im


@click.command()
@click.option('--budget', default=32768)
@click.option('--out', default='/iris/u/lucyshi/fpl_transfer/ktask/results/tasks.png')
def main(budget, out):
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.15])
    for i, s in enumerate((0, 1, 2)):
        b = base_eval(s)
        ax = fig.add_subplot(gs[0, i])
        im = draw(ax, grid({t: b[t]['success'].mean() for t in b}), f'base seed {s}: success '
                  f'(mean {np.mean([b[t]["success"].mean() for t in b]):.2f})', 'viridis', 0, 1)
    fig.colorbar(im, ax=ax, fraction=0.04)
    for i, k in enumerate((1, 4, 16)):
        acc = {t: [] for t in range(N_OBJ * N_TGT)}
        n = 0
        for s, c, kk, d, train in e1_runs():
            e, b = load_eval(f'{d}/eval_ep{budget}.json'), base_eval(s)
            if kk != k or e is None:
                continue
            n += 1
            for t in e:
                if t not in train:
                    acc[t].append(e[t]['success'].mean() - b[t]['success'].mean())
        ax = fig.add_subplot(gs[1, i])
        im = draw(ax, grid({t: np.mean(v) for t, v in acc.items() if v}),
                  f'K = {k}: mean change on tasks not trained ({n} runs)', 'RdBu', -0.3, 0.3, '{:+.2f}')
    fig.colorbar(im, ax=ax, fraction=0.04)
    ax = fig.add_subplot(gs[2, :])
    rows, labels, trained = [], [], []
    order = [o * N_TGT + t for o in range(N_OBJ) for t in range(N_TGT)]
    for s, c, k, d, train in sorted(e1_runs(), key=lambda r: (r[4][0] if r[2] == 1 else 99, r[0])):
        if k != 1:
            continue
        e, b = load_eval(f'{d}/eval_ep{budget}.json'), base_eval(s)
        if e is None:
            continue
        rows.append([e[t]['success'].mean() - b[t]['success'].mean() for t in order])
        labels.append(f'{task_name(train[0])} (seed {s})')
        trained.append(order.index(train[0]))
    M = np.array(rows)
    im = ax.imshow(M, cmap='RdBu', vmin=-0.4, vmax=0.4, aspect='auto')
    for r, j in enumerate(trained):
        ax.add_patch(plt.Rectangle((j - 0.5, r - 0.5), 1, 1, fill=False, ec='k', lw=2))
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f'{task_name(t)} [{SPLIT_OF[t]}]' for t in order], rotation=90, fontsize=7)
    for o in range(1, N_OBJ):
        ax.axvline(o * N_TGT - 0.5, color='k', lw=0.5)
    ax.set_title('K = 1: change on every task after PPO on one task (boxed: the trained task); columns grouped by object',
                 fontsize=9.5)
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
