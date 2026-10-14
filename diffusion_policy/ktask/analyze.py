"""
Analysis helpers for the K-task experiments: paired changes vs the base on task groups, with a two-level
bootstrap (runs, then start states within each task), and task relations to a training set.

  python -m ktask.analyze pilot --seed 1
  python -m ktask.analyze e1 [--out results/e1.md]
"""

import glob
import json
import os
import re

import click
import numpy as np

from ktask.data import SPLITS
from ktask.env import N_TASKS, task_name

ROOT = '/iris/u/lucyshi/fpl_transfer/ktask'
RUNS = f'{ROOT}/runs'


def load_eval(path):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    return {int(t): {k: np.asarray(v) for k, v in r.items()} for t, r in d['tasks'].items()}


def base_eval(s):
    return load_eval(f'{RUNS}/base/s{s}/eval.json')


def group_delta(pairs, tasks, metric='success', n_boot=2000, rng=None):
    """pairs: list of (eval, base_eval) runs. Mean over runs and tasks of the paired change on `tasks`,
    with a 95% CI from resampling runs and, within each run and task, start states."""
    rng = rng or np.random.RandomState(0)
    tasks = [t for t in tasks if all(t in e and t in b for e, b in pairs)]
    if not pairs or not tasks:
        return None
    diffs = [[(e[t][metric].astype(float) - b[t][metric].astype(float)) for t in tasks] for e, b in pairs]
    point = np.mean([np.mean([d.mean() for d in run]) for run in diffs])
    means = [np.mean([np.mean([b[t][metric].mean() for t in tasks]) for _, b in pairs]),
             np.mean([np.mean([e[t][metric].mean() for t in tasks]) for e, _ in pairs])]
    boots = np.empty(n_boot)
    for i in range(n_boot):
        runs = rng.randint(len(diffs), size=len(diffs))
        boots[i] = np.mean([np.mean([d[rng.randint(len(d), size=len(d))].mean() for d in diffs[r]]) for r in runs])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {'d': point, 'lo': lo, 'hi': hi, 'base': means[0], 'after': means[1], 'n_runs': len(pairs),
            'n_tasks': len(tasks)}


def fmt(r, base=True):
    if r is None:
        return '-'
    s = f"{r['after']:.2f} ({r['d']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}])"
    return s


def relation(task, train):
    """'both' if another training task shares the object and one shares the target, else 'object',
    'target' or 'neither'."""
    o, g = task // 4, task % 4
    so = any(t // 4 == o for t in train if t != task)
    sg = any(t % 4 == g for t in train if t != task)
    return 'both' if so and sg else 'object' if so else 'target' if sg else 'neither'


def ppo_curve(run_dir):
    """(episodes, mean training success) from ppo_log.txt."""
    xs, ys = [], []
    if not os.path.exists(f'{run_dir}/ppo_log.txt'):
        return xs, ys
    for line in open(f'{run_dir}/ppo_log.txt'):
        m = re.search(r'episodes (\d+): success ([\d.]+)', line)
        if m:
            xs.append(int(m.group(1)))
            ys.append(float(m.group(2)))
    return xs, ys


@click.group()
def cli():
    pass


@cli.command()
@click.option('--seed', default=1)
def pilot(seed):
    b = base_eval(seed)
    spec_dirs = sorted(glob.glob(f'{RUNS}/pilot/s{seed}/k*_lr*'))
    rows = []
    for d in spec_dirs:
        spec = json.load(open(f'{d}/ppo.json'))
        train = spec['tasks']
        xs, ys = ppo_curve(d)
        curve = ' '.join(f'{y:.2f}' for y in ys[::4])
        print(f"\n## {os.path.basename(d)}: train tasks {', '.join(task_name(t) for t in train[:4])}"
              f"{'...' if len(train) > 4 else ''}\ntraining success every 4 iterations: {curve}")
        held = [t for t in SPLITS['H']]
        rest = [t for t in SPLITS['R'] if t not in train]
        for ev in sorted(glob.glob(f'{d}/eval_ep*.json'), key=lambda p: int(re.search(r'ep(\d+)', p).group(1))):
            e = load_eval(ev)
            n = int(re.search(r'ep(\d+)', ev).group(1))
            cols = [('train', train), ('H', held), ('R-rest', rest), ('U', SPLITS['U'])]
            print(f'{n:6d} episodes: ' + ', '.join(f'{name} {fmt(group_delta([(e, b)], ts))}' for name, ts in cols if ts))


def e1_runs(episodes=65536):
    """[(seed, chain, K, run dir, training tasks)] for every E1 run with a ppo.json."""
    out = []
    for p in sorted(glob.glob(f'{RUNS}/e1/s*/k16*/ppo.json') + glob.glob(f'{RUNS}/e1/s*/c*/k*/ppo.json')):
        d = os.path.dirname(p)
        m = re.search(r'e1/s(\d+)/(?:c(\d+)/)?k(\d+)', d)
        spec = json.load(open(p))
        out.append((int(m.group(1)), int(m.group(2)) if m.group(2) else None, int(m.group(3)), d, spec['tasks']))
    return out


def e1_table(budget, metric='success'):
    """Rows per K: change on the training tasks, H, the rest of R, U; and on held-out tasks by relation."""
    rows = {}
    for s, c, k, d, train in e1_runs():
        e, b = load_eval(f'{d}/eval_ep{budget}.json'), base_eval(s)
        if e is None or b is None:
            continue
        rows.setdefault(k, []).append((e, b, train))
    lines = [f'| K | runs | training tasks | H (held out) | rest of R | U (never seen) | held out, shares object | '
             f'shares target | shares both | neither |', '|' + '---|' * 10]
    for k in sorted(rows):
        rs = rows[k]
        cells = [str(k), str(len(rs))]
        for group in ('train', 'H', 'rest', 'U'):
            per_run = []
            for e, b, train in rs:
                ts = {'train': train, 'H': SPLITS['H'], 'U': SPLITS['U'],
                      'rest': [t for t in SPLITS['R'] if t not in train]}[group]
                per_run.append((e, b, ts))
            cells.append(fmt_runs(per_run, metric))
        for rel in ('object', 'target', 'both', 'neither'):
            per_run = [(e, b, [t for t in SPLITS['H'] + [t for t in SPLITS['R'] if t not in train]
                               if relation(t, train) == rel]) for e, b, train in rs]
            cells.append(fmt_runs([r for r in per_run if r[2]], metric))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def fmt_runs(per_run, metric='success'):
    """Mean paired change over runs, each run with its own task list."""
    if not per_run:
        return '-'
    rng = np.random.RandomState(0)
    diffs = [[e[t][metric].astype(float) - b[t][metric].astype(float) for t in ts] for e, b, ts in per_run]
    point = np.mean([np.mean([d.mean() for d in run]) for run in diffs])
    boots = []
    for _ in range(1000):
        idx = rng.randint(len(diffs), size=len(diffs))
        boots.append(np.mean([np.mean([d[rng.randint(len(d), size=len(d))].mean() for d in diffs[r]]) for r in idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return f'{point:+.3f} [{lo:+.3f}, {hi:+.3f}]'


@cli.command()
@click.option('--budgets', default='4096,8192,16384,32768,65536')
@click.option('--out', default=None)
def e1(budgets, out):
    text = []
    for n in map(int, budgets.split(',')):
        text.append(f'\n### After {n} PPO episodes (change in success vs the base; 95% CI over runs and start states)\n')
        text.append(e1_table(n))
    text = '\n'.join(text)
    print(text)
    if out:
        open(out, 'w').write(text + '\n')


@cli.command()
@click.option('--seed', default=0)
def base(seed):
    b = base_eval(seed)
    for split in ('R', 'H', 'U'):
        print(f"{split}: mean success {np.mean([b[t]['success'].mean() for t in SPLITS[split]]):.3f}")
        print('  ' + ', '.join(f"{task_name(t)} {b[t]['success'].mean():.2f}" for t in SPLITS[split]))


if __name__ == '__main__':
    cli()
