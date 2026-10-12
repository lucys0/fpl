"""
Aggregate scripts/eval_composition.py results over seeds into a Table-6-style
summary (mean +/- standard error over seeds) and a Fig.-4-style plot of time to
success per peg, next to the demonstrations' own times.

Expects <eval_dir>/<method>_seed<N>/results.json.

Usage:
  python scripts/summarize_composition.py --eval_dir <dir> \
      --demo_hdf5 shared_data_slow_fast/scripted_data/demos.hdf5 --out_dir <dir>
"""

import collections
import glob
import json
import os
import re

import click
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# FPL paper, Table 6 (Bimodal Square, standard direction, mean +/- SE over 3 seeds)
PAPER_TABLE6 = {
    'fpl': {'success': (1.00, 0.00), 't1s': (253.13, 0.01), 'throughput': (1.19, 0.00)},
    'single_pref': {'success': (0.89, 0.11), 't1s': (395.93, 1.80), 'throughput': (0.67, 0.08)},
}
# The condition each method is scored under for Table 6 (launch_*.sh EVAL_Z_POSITIVE)
TABLE6_CONDITION = {'fpl': 'fast_right', 'single_pref': 'high'}
METRICS = ['right_peg_success', 'left_peg_success', 't1s_right', 't1s_left',
           'throughput_right', 'throughput_left']


def mean_se(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None, 0
    a = np.asarray(values, dtype=float)
    se = a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
    return float(a.mean()), float(se), len(a)


def fmt(m, se, digits=2):
    return 'n/a' if m is None else f'{m:.{digits}f} ± {se:.{digits}f}'


def demo_steps(demo_hdf5):
    """Steps to first success of each scripted demo, per peg (demos end at success)."""
    steps = collections.defaultdict(list)
    with h5py.File(demo_hdf5, 'r') as f:
        for demo in f['data'].values():
            steps[demo.attrs['target_peg']].append(int(demo['actions'].shape[0]))
    return {peg: np.asarray(v) for peg, v in steps.items()}


@click.command()
@click.option('--eval_dir', required=True)
@click.option('--demo_hdf5', required=True)
@click.option('--out_dir', required=True)
def main(eval_dir, demo_hdf5, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    runs = collections.defaultdict(dict)   # method -> seed -> results
    for path in sorted(glob.glob(os.path.join(eval_dir, '*_seed*', 'results.json'))):
        m = re.match(r'(.+)_seed(\w+)$', os.path.basename(os.path.dirname(path)))
        with open(path) as f:
            runs[m.group(1)][m.group(2)] = json.load(f)
    demos = demo_steps(demo_hdf5)
    fastest_right_demo = int(demos['right'].min())

    lines = ['# Bimodal Square composition eval', '']
    lines.append(f"Demos: {len(demos['left'])} left-peg, time to success "
                 f"{demos['left'].min()}–{demos['left'].max()} (median {np.median(demos['left']):.0f}); "
                 f"{len(demos['right'])} right-peg, {demos['right'].min()}–{demos['right'].max()} "
                 f"(median {np.median(demos['right']):.0f}).")
    lines.append('')

    # Table 6 comparison
    lines += ['## Table 6 (standard direction: success = nut on the right peg)', '',
              '| Method | Condition | Seeds | Success | T1S | Throughput | Paper success | Paper T1S | Paper throughput |',
              '|---|---|---|---|---|---|---|---|---|']
    summary = {'demos': {peg: v.tolist() for peg, v in demos.items()}, 'methods': {}}
    for method, seeds in sorted(runs.items()):
        cond = TABLE6_CONDITION.get(method)
        if cond is None:
            continue
        s, s_se, n = mean_se([r['conditions'][cond]['summary']['right_peg_success'] for r in seeds.values()])
        t, t_se, _ = mean_se([r['conditions'][cond]['summary']['t1s_right'] for r in seeds.values()])
        h, h_se, _ = mean_se([r['conditions'][cond]['summary']['throughput_right'] for r in seeds.values()])
        p = PAPER_TABLE6.get(method, {})
        paper = [fmt(*p[k]) if k in p else 'n/a' for k in ('success', 't1s', 'throughput')]
        lines.append(f'| {method} | {cond} | {n} | {fmt(s, s_se)} | {fmt(t, t_se, 1)} | {fmt(h, h_se)} | '
                     + ' | '.join(paper) + ' |')
    lines.append('')

    # Every condition, both pegs
    lines += ['## All conditions (mean ± SE over seeds)', '',
              '| Method | Condition | Target | Right-peg success | Left-peg success | T1S right | T1S left | '
              f'Right placements faster than every right demo (<{fastest_right_demo} steps) |',
              '|---|---|---|---|---|---|---|---|']
    for method, seeds in sorted(runs.items()):
        summary['methods'][method] = {}
        conds = next(iter(seeds.values()))['conditions']
        for cond, info in conds.items():
            per_seed = [r['conditions'][cond] for r in seeds.values()]
            stats = {k: mean_se([c['summary'][k] for c in per_seed]) for k in METRICS}
            right_steps = [s for c in per_seed for s in c['first_success_step_right']]
            n_rollouts = sum(r['n_rollouts'] for r in seeds.values())
            faster = sum(s < fastest_right_demo for s in right_steps)
            summary['methods'][method][cond] = {
                'target': info['target'], 'seeds': sorted(seeds),
                **{k: {'mean': v[0], 'se': v[1]} for k, v in stats.items()},
                'right_faster_than_demos': faster, 'n_rollouts_total': n_rollouts,
                'first_success_step_left': [s for c in per_seed for s in c['first_success_step_left']],
                'first_success_step_right': right_steps,
            }
            lines.append(f"| {method} | {cond} | {info['target']} | {fmt(*stats['right_peg_success'][:2])} | "
                         f"{fmt(*stats['left_peg_success'][:2])} | {fmt(*stats['t1s_right'][:2], 1)} | "
                         f"{fmt(*stats['t1s_left'][:2], 1)} | {faster}/{n_rollouts} rollouts |")
    lines.append('')

    # Fig. 4 style: time to success per peg, demos next to each method/condition
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, peg in zip(axes, ('left', 'right')):
        labels, data = [f'demos\n(n={len(demos[peg])})'], [demos[peg]]
        for method, conds in sorted(summary['methods'].items()):
            for cond, info in conds.items():
                steps = info[f'first_success_step_{peg}']
                if steps:
                    labels.append(f'{method}\n{cond}\n(n={len(steps)})')
                    data.append(np.asarray(steps))
        ax.boxplot(data, widths=0.6, showfliers=False)
        for i, d in enumerate(data, start=1):
            ax.scatter(np.random.normal(i, 0.06, len(d)), d, s=6, alpha=0.4)
        ax.axhline(fastest_right_demo, ls='--', c='gray', lw=1)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(f'{peg} peg')
        ax.set_ylabel('time to success (steps)')
    fig.suptitle(f'Time to success per peg (dashed: fastest right-peg demo, {fastest_right_demo} steps)')
    fig.tight_layout()
    fig_path = os.path.join(out_dir, 'time_to_success_per_peg.png')
    fig.savefig(fig_path, dpi=150)

    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, 'summary.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'Figure: {fig_path}')


if __name__ == '__main__':
    main()
