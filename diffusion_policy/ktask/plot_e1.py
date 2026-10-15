"""
E1 figure: does PPO on K tasks help held-out tasks, and does that change with K?

  (a) change on the K training tasks vs K (final budget)
  (b) change on the fixed held-out set H vs K, per-run dots
  (c) (training-task change, H change) along each run's checkpoints, one color per K
  (d) held-out change by relation to the training set (shares the object / target / both / neither) vs K
  (e) change on the never-seen tasks U (zero-shot) vs K

  python -m ktask.plot_e1 [--budget N] [--out results/e1.png]
"""

import click
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ktask.analyze import RUNS, base_eval, e1_runs, load_eval, relation
from ktask.data import SPLITS

COLORS = {1: 'tab:blue', 2: 'tab:orange', 4: 'tab:green', 8: 'tab:red', 16: 'tab:purple'}


def mean_delta(e, b, tasks, metric='success'):
    tasks = [t for t in tasks if t in e and t in b]
    if not tasks:
        return np.nan
    return float(np.mean([e[t][metric].mean() - b[t][metric].mean() for t in tasks]))


def groups(train):
    rest = [t for t in SPLITS['R'] if t not in train]
    held = SPLITS['H'] + rest
    out = {'train': train, 'H': SPLITS['H'], 'rest': rest, 'U': SPLITS['U']}
    for rel in ('object', 'target', 'both', 'neither'):
        out[rel] = [t for t in held if relation(t, train) == rel]
    return out


def train_delta(s, c, k, d, train, budget, e, b):
    """Training-task change; for K = 1 the precise E1x evaluation (512 episodes) when it exists."""
    if k == 1:
        x, xb = load_eval(f'{RUNS}/e1x/s{s}/c{c}/ep{budget}.json'), load_eval(f'{RUNS}/e1x/s{s}/c{c}/base.json')
        if x is not None and xb is not None:
            t = train[0]
            return float(x[t]['success'].mean() - xb[t]['success'].mean())
    return mean_delta(e, b, train)


def run_paths(budgets):
    """[(K, [(d_train, d_H, d_U) per budget])] for runs evaluated at every budget."""
    out = []
    for s, c, k, d, train in e1_runs():
        b = base_eval(s)
        pts = []
        for n in budgets:
            e = load_eval(f'{d}/eval_ep{n}.json')
            if e is None or b is None:
                break
            pts.append((train_delta(s, c, k, d, train, n, e, b), mean_delta(e, b, SPLITS['H']),
                        mean_delta(e, b, SPLITS['U'])))
        if len(pts) == len(budgets):
            out.append((k, pts))
    return out


def summary(budget):
    """{K: [ {group: delta} per run ]} at one budget."""
    out = {}
    for s, c, k, d, train in e1_runs():
        e, b = load_eval(f'{d}/eval_ep{budget}.json'), base_eval(s)
        if e is None or b is None:
            continue
        r = {g: mean_delta(e, b, ts) for g, ts in groups(train).items()}
        r['train'] = train_delta(s, c, k, d, train, budget, e, b)
        out.setdefault(k, []).append(r)
    return out


def errbar(ax, ks, vals, color='k', label=None):
    m = [np.nanmean(v) if len(v) else np.nan for v in vals]
    se = [np.nanstd(v) / np.sqrt(max(1, np.sum(~np.isnan(v)))) * 1.96 if len(v) > 1 else 0 for v in vals]
    ax.errorbar(ks, m, yerr=se, fmt='-o', color=color, capsize=3, label=label)
    for k, v in zip(ks, vals):
        ax.scatter(np.full(len(v), k) * np.exp(np.random.RandomState(k).uniform(-0.06, 0.06, len(v))), v,
                   s=10, color=color, alpha=0.35)


@click.command()
@click.option('--budget', default=None, type=int, help='episodes (default: the largest with results)')
@click.option('--budgets', default='4096,8192,16384,32768')
@click.option('--out', default='/iris/u/lucyshi/fpl_transfer/ktask/results/e1.png')
def main(budget, budgets, out):
    budgets = [int(b) for b in budgets.split(',')]
    if budget is None:
        budget = max([b for b in budgets if summary(b)] or [budgets[0]])
    S = summary(budget)
    ks = sorted(S)
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    ax = axes[0, 0]
    errbar(ax, ks, [np.array([r['train'] for r in S[k]]) for k in ks])
    ax.set_title(f'(a) Training tasks (after {budget} PPO episodes in total)', fontsize=10)
    ax = axes[0, 1]
    errbar(ax, ks, [np.array([r['H'] for r in S[k]]) for k in ks], label='H (fixed held-out set)')
    errbar(ax, [k for k in ks if k < 16], [np.array([r['rest'] for r in S[k]]) for k in ks if k < 16], color='tab:gray',
           label='rest of R (not trained)')
    ax.legend(fontsize=8)
    ax.set_title('(b) Held-out tasks (in pretraining, never trained with RL)', fontsize=10)
    ax = axes[0, 2]
    paths = run_paths(budgets)
    for k in ks:
        P = np.array([p for kk, p in paths if kk == k])            # runs x budgets x 3
        if len(P):
            xs = [0.0] + list(P[:, :, 0].mean(0))
            ys = [0.0] + list(P[:, :, 1].mean(0))
            ax.plot(xs, ys, '-o', color=COLORS.get(k, 'k'), label=f'K = {k} ({len(P)} runs)', ms=4)
    lim = max(0.05, max([abs(x) for _, p in paths for x, _, _ in p] or [0.1]))
    ax.plot([0, lim], [0, lim], color='k', lw=0.6, alpha=0.4)
    ax.axhline(0, color='k', lw=0.6)
    ax.axvline(0, color='k', lw=0.6)
    ax.set_xlabel('change on training tasks')
    ax.set_ylabel('change on H')
    ax.legend(fontsize=8)
    ax.set_title(f'(c) Held-out vs training change along training\n(checkpoints {budgets}; runs with all of them)', fontsize=10)
    ax = axes[1, 0]
    for rel, c in (('object', 'tab:blue'), ('target', 'tab:orange'), ('both', 'tab:green'), ('neither', 'tab:red')):
        vals = [np.array([r[rel] for r in S[k] if not np.isnan(r[rel])]) for k in ks]
        kk = [k for k, v in zip(ks, vals) if len(v)]
        errbar(ax, kk, [v for v in vals if len(v)], color=c, label=f'shares {rel}' if rel != 'neither' else 'shares neither')
    ax.legend(fontsize=8)
    ax.set_title('(d) Held-out tasks by relation to the training set', fontsize=10)
    ax = axes[1, 1]
    errbar(ax, ks, [np.array([r['U'] for r in S[k]]) for k in ks])
    ax.set_title('(e) Never-seen tasks U (zero-shot)', fontsize=10)
    ax = axes[1, 2]
    for j, (name, color) in enumerate((('H', 'k'), ('U', 'tab:gray'))):
        vals = []
        for k in ks:
            sl = []
            for kk, p in paths:
                if kk == k:
                    x = np.array([q[0] for q in p])
                    y = np.array([q[1 + j] for q in p])
                    if (x ** 2).sum() > 1e-4:
                        sl.append(float((x * y).sum() / (x ** 2).sum()))
            vals.append(np.array(sl))
        kk = [k for k, v in zip(ks, vals) if len(v)]
        errbar(ax, kk, [v for v in vals if len(v)], color=color, label=f'{name} per unit of training change')
    ax.axhline(1, color='k', lw=0.6, ls=':')
    ax.legend(fontsize=8)
    ax.set_title('(f) Transfer per unit of training improvement\n(per-run slope through the origin over checkpoints)', fontsize=10)
    for ax in list(axes[0]) + list(axes[1]):
        if ax is not axes[0, 2]:
            ax.set_xscale('log', base=2)
            ax.set_xticks([1, 2, 4, 8, 16])
            ax.set_xticklabels([1, 2, 4, 8, 16])
            ax.axhline(0, color='k', lw=0.6)
            ax.set_xlabel('K (tasks trained with PPO; same total episodes)')
            ax.set_ylabel('change in success vs base')
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
