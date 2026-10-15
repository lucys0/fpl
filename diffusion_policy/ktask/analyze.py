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


def phase_table(budget):
    """Held-out tasks (H and the rest of R): change in P(lifted) and in P(success | lifted) per K."""
    rows = {}
    for s, c, k, d, train in e1_runs():
        e, b = load_eval(f'{d}/eval_ep{budget}.json'), base_eval(s)
        if e is None or b is None:
            continue
        held = SPLITS['H'] + [t for t in SPLITS['R'] if t not in train]
        out = {}
        for name, ts in (('train', train), ('held', held), ('U', SPLITS['U'])):
            dl = np.mean([e[t]['lifted'].mean() - b[t]['lifted'].mean() for t in ts])
            cond = lambda r: r['success'][r['lifted']].mean() if r['lifted'].any() else np.nan
            dc = np.nanmean([cond(e[t]) - cond(b[t]) for t in ts])
            out[name] = (dl, dc)
        rows.setdefault(k, []).append(out)
    lines = ['| K | runs | training: lifted | training: success if lifted | held out: lifted | '
             'held out: success if lifted | U: lifted | U: success if lifted |', '|' + '---|' * 8]
    for k in sorted(rows):
        cells = [str(k), str(len(rows[k]))]
        for name in ('train', 'held', 'U'):
            for i in (0, 1):
                cells.append(f"{np.nanmean([r[name][i] for r in rows[k]]):+.3f}")
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def speed_table(budget):
    """Relative change in steps to success, on start states where both the base and the run succeed."""
    rows = {}
    for s, c, k, d, train in e1_runs():
        e, b = load_eval(f'{d}/eval_ep{budget}.json'), base_eval(s)
        if e is None or b is None:
            continue
        held = SPLITS['H'] + [t for t in SPLITS['R'] if t not in train]
        out = {}
        for name, ts in (('train', train), ('held', held), ('U', SPLITS['U'])):
            rel = []
            for t in ts:
                both = e[t]['success'] & b[t]['success']
                if both.sum() >= 5:
                    rel.append(e[t]['steps'][both].mean() / b[t]['steps'][both].mean() - 1)
            out[name] = np.mean(rel) if rel else np.nan
        rows.setdefault(k, []).append(out)
    lines = ['| K | runs | training tasks | held out (H + rest of R) | U |', '|---|---|---|---|---|']
    for k in sorted(rows):
        lines.append(f'| {k} | {len(rows[k])} | ' + ' | '.join(
            f"{100 * np.nanmean([r[n] for r in rows[k]]):+.1f}%" for n in ('train', 'held', 'U')) + ' |')
    return '\n'.join(lines)


@cli.command()
@click.option('--budgets', default='4096,8192,16384,32768,65536')
@click.option('--out', default=None)
def e1(budgets, out):
    text = []
    for n in map(int, budgets.split(',')):
        text.append(f'\n### After {n} PPO episodes (change in success vs the base; 95% CI over runs and start states)\n')
        text.append(e1_table(n))
        text.append(f'\nWhere the change comes from (mean over runs): grasp (lifted) vs after the grasp\n')
        text.append(phase_table(n))
        text.append(f'\nSpeed: change in steps to success where both the base and the run succeed (mean over runs)\n')
        text.append(speed_table(n))
    text = '\n'.join(text)
    print(text)
    if out:
        open(out, 'w').write(text + '\n')


E2_ARMS = (('ppo', 'PPO', [4096, 8192, 16384, 32768], 'eval_ep{}.json'),
           ('filtered_bc', "filtered BC on PPO's episodes", [1000, 3000, 6000, 10000], 'eval_step{}.json'),
           ('sft_expert', 'SFT on clean expert demos', [1000, 3000, 6000, 10000], 'eval_step{}.json'))


def e2_points(k):
    """{arm: [(point label, [per-run dicts of group deltas])]} for E2 sources with K tasks."""
    out = {}
    for s, c, kk, d, train in e1_runs():
        if kk != k or not os.path.exists(f'{d}/ppo.json') or not json.load(open(f'{d}/ppo.json')).get('log_rollouts'):
            continue
        tag = 'k16' if k == 16 else f'c{c}/k{k}'
        b = base_eval(s)
        for arm, _, points, pat in E2_ARMS:
            run_dir = d if arm == 'ppo' else f'{RUNS}/e2/s{s}/{tag}/{arm}'
            for p in points:
                e = load_eval(f'{run_dir}/{pat.format(p)}')
                if e is None or b is None:
                    continue
                held = SPLITS['H'] + [t for t in SPLITS['R'] if t not in train]
                g = {'train': train, 'held': held, 'H': SPLITS['H'], 'U': SPLITS['U']}
                out.setdefault(arm, {}).setdefault(p, []).append(
                    {n: float(np.mean([e[t]['success'].mean() - b[t]['success'].mean() for t in ts])) for n, ts in g.items()})
    return out


@cli.command()
@click.option('--out', default=None)
def e2(out):
    lines = []
    for k in (1, 4, 16):
        pts = e2_points(k)
        if not pts:
            continue
        lines.append(f'\n### K = {k}: change vs the base (mean over runs)\n')
        lines.append('| arm | point | runs | training tasks | held out (H + rest of R) | H | U |')
        lines.append('|---|---|---|---|---|---|---|')
        for arm, label, points, _ in E2_ARMS:
            for p in points:
                rs = pts.get(arm, {}).get(p)
                if not rs:
                    continue
                m = {n: np.mean([r[n] for r in rs]) for n in ('train', 'held', 'H', 'U')}
                unit = 'episodes' if arm == 'ppo' else 'steps'
                lines.append(f"| {label} | {p} {unit} | {len(rs)} | {m['train']:+.3f} | {m['held']:+.3f} | "
                             f"{m['H']:+.3f} | {m['U']:+.3f} |")
    text = '\n'.join(lines)
    print(text)
    if out:
        open(out, 'w').write(text + '\n')


@cli.command()
@click.option('--episodes', default=32768)
@click.option('--out', default=None)
def e3(episodes, out):
    """Learning the never-seen tasks U from k clean demos each, from the base and from PPO on K tasks:
    U success before (zero-shot) and after, and H after (kept), paired against the base start."""
    starts = (('base', 'base'), ('ppo_k1', 'PPO on 1 task'), ('ppo_k4', 'PPO on 4 tasks'), ('ppo_k16', 'PPO on 16 tasks'))
    lines = ['| start | U zero-shot | U after 5 demos | vs base start | U after 20 demos | vs base start | '
             'H after 20 demos | vs base start |', '|---|---|---|---|---|---|---|---|']
    for key, label in starts:
        zs, cells = [], {5: [], 20: []}
        held = []
        for s in (0, 1, 2):
            b0 = base_eval(s)
            if key == 'base':
                z = b0
            else:
                k = int(key.split('k')[1])
                from ktask.experiments import e1_dir
                z = load_eval(f'{e1_dir(s, 0, k)}/eval_ep{episodes}.json')
            if z is not None and b0 is not None:
                zs.append(np.mean([z[t]['success'].mean() for t in SPLITS['U']]))
            for kd in (5, 20):
                e = load_eval(f'{RUNS}/e3/s{s}/{key}/k{kd}/eval.json')
                eb = load_eval(f'{RUNS}/e3/s{s}/base/k{kd}/eval.json')
                if e is None or eb is None:
                    continue
                cells[kd].append((np.mean([e[t]['success'].mean() for t in SPLITS['U']]),
                                  np.mean([e[t]['success'].mean() - eb[t]['success'].mean() for t in SPLITS['U']])))
                if kd == 20:
                    held.append((np.mean([e[t]['success'].mean() for t in SPLITS['H']]),
                                 np.mean([e[t]['success'].mean() - eb[t]['success'].mean() for t in SPLITS['H']])))
        f = lambda v, i: f'{np.mean([x[i] for x in v]):.2f}' if v and i == 0 else (f'{np.mean([x[i] for x in v]):+.3f}' if v else '-')
        lines.append(f"| {label} | {np.mean(zs):.2f} (n={len(zs)}) | {f(cells[5], 0)} | {f(cells[5], 1)} | "
                     f"{f(cells[20], 0)} | {f(cells[20], 1)} | {f(held, 0)} | {f(held, 1)} |" if zs else f'| {label} | - |')
    text = '\n'.join(lines)
    print(text)
    if out:
        open(out, 'w').write(text + '\n')


@cli.command()
def e1x():
    """Precise (512-episode) training-task change of each K = 1 run, vs the 128-episode main evaluation."""
    for s, c, k, d, train in e1_runs():
        if k != 1:
            continue
        xb = load_eval(f'{RUNS}/e1x/s{s}/c{c}/base.json')
        if xb is None:
            continue
        t = train[0]
        cells = []
        for ep in (4096, 8192, 16384, 32768):
            x = load_eval(f'{RUNS}/e1x/s{s}/c{c}/ep{ep}.json')
            e, b = load_eval(f'{d}/eval_ep{ep}.json'), base_eval(s)
            precise = f"{x[t]['success'].mean() - xb[t]['success'].mean():+.3f}" if x else '-'
            main = f"{e[t]['success'].mean() - b[t]['success'].mean():+.3f}" if e else '-'
            cells.append(f'{precise} ({main})')
        print(f"s{s} c{c} {task_name(t):18s} base {xb[t]['success'].mean():.2f}: 4k/8k/16k/32k "
              + ' '.join(cells) + '   [512 episodes (128-episode main eval)]')


@cli.command()
@click.option('--seed', default=0)
def base(seed):
    b = base_eval(seed)
    for split in ('R', 'H', 'U'):
        print(f"{split}: mean success {np.mean([b[t]['success'].mean() for t in SPLITS[split]]):.3f}")
        print('  ' + ', '.join(f"{task_name(t)} {b[t]['success'].mean():.2f}" for t in SPLITS[split]))


if __name__ == '__main__':
    cli()
