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
# v2 (default): evaluations and PPO after the observation-history fix; KTASK_VERSION=v1 reproduces the
# earlier numbers (rollouts with observations 8 steps apart)
VERSION = os.environ.get('KTASK_VERSION', 'v2')
SUFFIX = '' if VERSION == 'v1' else VERSION


def load_eval(path):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    return {int(t): {k: np.asarray(v) for k, v in r.items()} for t, r in d['tasks'].items()}


def base_eval(s):
    return load_eval(f'{RUNS}/base/s{s}/eval{"_" + SUFFIX if SUFFIX else ""}.json')


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
    for p in sorted(glob.glob(f'{RUNS}/e1{SUFFIX}/s*/k16*/ppo.json') + glob.glob(f'{RUNS}/e1{SUFFIX}/s*/c*/k*/ppo.json')):
        d = os.path.dirname(p)
        m = re.search(r'e1(?:v\d)?/s(\d+)/(?:c(\d+)/)?k(\d+)', d)
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
@click.option('--out', default=None)
def slopes(out):
    """Held-out change per unit of training-task change: per run, the least-squares slope through the origin
    over its checkpoints (4k-32k episodes); mean over runs per K with a 95% bootstrap CI over runs."""
    from ktask.plot_e1 import run_paths
    paths = run_paths([4096, 8192, 16384, 32768])
    rng = np.random.RandomState(0)
    lines = ['| K | runs | H per unit of training change | U per unit of training change | runs with H slope > 0.5 |',
             '|---|---|---|---|---|']
    for k in sorted({k for k, _ in paths}):
        sl = {0: [], 1: []}
        for kk, p in paths:
            if kk != k:
                continue
            x = np.array([q[0] for q in p])
            if (x ** 2).sum() < 1e-4:
                continue
            for j in (0, 1):
                y = np.array([q[1 + j] for q in p])
                sl[j].append(float((x * y).sum() / (x ** 2).sum()))
        cells = []
        for j in (0, 1):
            v = np.array(sl[j])
            b = [rng.choice(v, len(v)).mean() for _ in range(2000)]
            cells.append(f'{v.mean():.2f} [{np.percentile(b, 2.5):.2f}, {np.percentile(b, 97.5):.2f}]')
        lines.append(f'| {k} | {len(sl[0])} | {cells[0]} | {cells[1]} | {np.sum(np.array(sl[0]) > 0.5)}/{len(sl[0])} |')
    text = '\n'.join(lines)
    print(text)
    if out:
        open(out, 'w').write(text + '\n')


@cli.command()
@click.option('--out', default=None)
def per_task(out):
    """E1 checkpoints regrouped by episodes per task (n / K): at equal data per task, larger K has more
    iterations (n / 1024) and more total episodes."""
    from ktask.plot_e1 import mean_delta, train_delta
    rows = {}
    for s, c, k, d, train in e1_runs():
        b = base_eval(s)
        for n in (4096, 8192, 16384, 32768):
            e = load_eval(f'{d}/eval_ep{n}.json')
            if e is None or b is None:
                continue
            rows.setdefault((n // k, k), []).append((train_delta(s, c, k, d, train, n, e, b),
                                                     mean_delta(e, b, SPLITS['H']), mean_delta(e, b, SPLITS['U'])))
    lines = ['| episodes per task | K | PPO iterations | runs | training tasks | H | U |', '|---|---|---|---|---|---|---|']
    for (per, k) in sorted(rows, key=lambda x: (-x[0], x[1])):
        if per < 1024:
            continue
        v = np.array(rows[(per, k)])
        lines.append(f'| {per} | {k} | {per * k // 1024} | {len(v)} | {v[:, 0].mean():+.3f} | {v[:, 1].mean():+.3f} | '
                     f'{v[:, 2].mean():+.3f} |')
    text = '\n'.join(lines)
    print(text)
    if out:
        open(out, 'w').write(text + '\n')


def _deltas(e, b, train):
    held = SPLITS['H'] + [t for t in SPLITS['R'] if t not in train]
    f = lambda ts: float(np.mean([e[t]['success'].mean() - b[t]['success'].mean() for t in ts]))
    return f(train), f(held), f(SPLITS['U'])


@cli.command()
def e2x():
    """Filtered-BC diagnosis: for each K = 1 run, training-task and held-out change (10k steps) of BC on
    different data, next to PPO and the E2 arms."""
    from ktask.experiments import E2X_RUNS, chain, e1_dir
    arms = [('PPO (32k episodes)', lambda s, c: f'{e1_dir(s, c, 1)}/eval_ep32768.json'),
            ("filtered BC, PPO's logged episodes", lambda s, c: f'{RUNS}/e2/s{s}/c{c}/k1/filtered_bc/eval_step10000.json'),
            ('filtered BC, final PPO, no chain noise', lambda s, c: f'{RUNS}/e2x/s{s}/c{c}/fbc_det/eval_step10000.json'),
            ('filtered BC, final PPO, chain noise', lambda s, c: f'{RUNS}/e2x/s{s}/c{c}/fbc_sigma/eval_step10000.json'),
            ("filtered BC, the base's own episodes", lambda s, c: f'{RUNS}/e2x/s{s}/c{c}/fbc_self/eval_step10000.json'),
            ("filtered BC, logged episodes + replay", lambda s, c: f'{RUNS}/e2x/s{s}/c{c}/fbc_logged_replay/eval_step10000.json'),
            ('SFT on expert demos', lambda s, c: f'{RUNS}/e2/s{s}/c{c}/k1/sft_expert/eval_step10000.json'),
            ('SFT on expert demos + replay', lambda s, c: f'{RUNS}/e2x/s{s}/c{c}/sft_expert_replay/eval_step10000.json')]
    head = ' | '.join(f'{chain(s, c)[0]}: train, held out' for s, c in E2X_RUNS)
    from ktask.env import task_name
    print('| arm | ' + ' | '.join(f'{task_name(chain(s, c)[0])} (seed {s}): training, held out' for s, c in E2X_RUNS)
          + ' | mean training | mean held out |')
    print('|---' * (len(E2X_RUNS) + 3) + '|')
    for label, path in arms:
        cells, tr, ho = [], [], []
        for s, c in E2X_RUNS:
            e, b = load_eval(path(s, c)), base_eval(s)
            if e is None:
                cells.append('-')
                continue
            a, h, _ = _deltas(e, b, [chain(s, c)[0]])
            cells.append(f'{a:+.2f}, {h:+.2f}')
            tr.append(a)
            ho.append(h)
        m = f'{np.mean(tr):+.3f} | {np.mean(ho):+.3f}' if tr else '- | -'
        print(f'| {label} | ' + ' | '.join(cells) + f' | {m} |')


@cli.command()
def e3x():
    """Learnability curves: U success after SFT on k demos per U task for 0 / 250 / 1000 / 4000 steps (lr 2e-5),
    per start; change vs the base start at the same point."""
    from ktask.experiments import e1_dir
    starts = ('base', 'ppo_k1', 'ppo_k4', 'ppo_k16', 'sft_expert_k16')
    zero = {'base': lambda s: f'{RUNS}/base/s{s}/eval.json',
            'ppo_k1': lambda s: f'{e1_dir(s, 0, 1)}/eval_ep32768.json',
            'ppo_k4': lambda s: f'{e1_dir(s, 0, 4)}/eval_ep32768.json',
            'ppo_k16': lambda s: f'{e1_dir(s, 0, 16)}/eval_ep32768.json',
            'sft_expert_k16': lambda s: f'{RUNS}/e2/s{s}/k16/sft_expert/eval_step10000.json'}
    U = SPLITS['U']
    for kd in (1, 5):
        print(f'\n### {kd} demo(s) per never-seen task: U success (change vs the base start at the same point)\n')
        print('| start | 0 steps | 250 | 1000 | 4000 | H after 4000 |')
        print('|---|---|---|---|---|---|')
        for st in starts:
            cells = []
            for point in (0, 250, 1000, 4000):
                vals, dv = [], []
                for s in (0, 1, 2):
                    if point == 0:
                        e, eb = load_eval(zero[st](s)), load_eval(zero['base'](s))
                    else:
                        e = load_eval(f'{RUNS}/e3x/s{s}/{st}/k{kd}/eval_step{point}.json')
                        eb = load_eval(f'{RUNS}/e3x/s{s}/base/k{kd}/eval_step{point}.json')
                    if e is None or eb is None:
                        continue
                    vals.append(np.mean([e[t]['success'].mean() for t in U]))
                    dv.append(np.mean([e[t]['success'].mean() - eb[t]['success'].mean() for t in U]))
                cells.append(f'{np.mean(vals):.2f} ({np.mean(dv):+.3f}, n={len(vals)})' if vals else '-')
            hv = []
            for s in (0, 1, 2):
                e = load_eval(f'{RUNS}/e3x/s{s}/{st}/k{kd}/eval_step4000.json')
                if e is not None:
                    hv.append(np.mean([e[t]['success'].mean() for t in SPLITS['H']]))
            print(f'| {st} | ' + ' | '.join(cells) + f" | {np.mean(hv):.2f} |" if hv else f'| {st} | ' + ' | '.join(cells) + ' | - |')


@cli.command()
def e3r():
    """RL learnability: U success during PPO on the 8 U tasks, per start (0 = before RL on U)."""
    from ktask.experiments import e1_dir
    zero = {'base': lambda s: f'{RUNS}/base/s{s}/eval.json',
            'ppo_k16': lambda s: f'{e1_dir(s, 0, 16)}/eval_ep32768.json',
            'sft_expert_k16': lambda s: f'{RUNS}/e2/s{s}/k16/sft_expert/eval_step10000.json'}
    U = SPLITS['U']
    print('| start | 0 episodes | 2048 | 4096 | 8192 | 16384 | H at 16384 |')
    print('|---|---|---|---|---|---|---|')
    for st in ('base', 'ppo_k16', 'sft_expert_k16'):
        cells = []
        for n in (0, 2048, 4096, 8192, 16384):
            vals = []
            for s in (0, 1, 2):
                e = load_eval(zero[st](s)) if n == 0 else load_eval(f'{RUNS}/e3r/s{s}/{st}/eval_ep{n}.json')
                if e is not None:
                    vals.append(np.mean([e[t]['success'].mean() for t in U]))
            cells.append(f'{np.mean(vals):.2f} (n={len(vals)})' if vals else '-')
        hv = [np.mean([e[t]['success'].mean() for t in SPLITS['H']]) for s in (0, 1, 2)
              for e in [load_eval(f'{RUNS}/e3r/s{s}/{st}/eval_ep16384.json')] if e is not None]
        print(f'| {st} | ' + ' | '.join(cells) + (f' | {np.mean(hv):.2f} |' if hv else ' | - |'))


@cli.command()
def e1p():
    """Equal data per task (2,048) and equal iterations (32): change at each checkpoint per K; K = 16 is E1."""
    from ktask.experiments import chain, e1_dir
    from ktask.plot_e1 import mean_delta
    print('| K | slots per iteration | runs | iteration 4 (train, H) | 8 | 16 | 32 | U at 32 | H per unit of training change |')
    print('|---|---|---|---|---|---|---|---|---|')
    for k in (1, 2, 4, 8, 16):
        per_it = {4: [], 8: [], 16: [], 32: []}
        u32, slopes = [], []
        for s in (0, 1, 2):
            for c in ((0, 1) if k < 16 else (0, 1)):
                d = f'{RUNS}/e1p/s{s}/c{c}/k{k}' if k < 16 else e1_dir(s, c, 16)
                n_envs = 64 * k
                b = base_eval(s)
                train = chain(s, c)[:k] if k < 16 else SPLITS['R']
                xs, ys = [], []
                for it in (4, 8, 16, 32):
                    e = load_eval(f'{d}/eval_ep{n_envs * it}.json')
                    if e is None:
                        continue
                    a, h = mean_delta(e, b, train), mean_delta(e, b, SPLITS['H'])
                    per_it[it].append((a, h))
                    xs.append(a)
                    ys.append(h)
                    if it == 32:
                        u32.append(mean_delta(e, b, SPLITS['U']))
                x, y = np.array(xs), np.array(ys)
                if len(x) == 4 and (x ** 2).sum() > 1e-4:
                    slopes.append((x * y).sum() / (x ** 2).sum())
        cells = [f'{np.mean([v[0] for v in per_it[it]]):+.3f}, {np.mean([v[1] for v in per_it[it]]):+.3f}'
                 if per_it[it] else '-' for it in (4, 8, 16, 32)]
        print(f'| {k} | {64 * k} | {len(per_it[32])} | ' + ' | '.join(cells) +
              f" | {np.mean(u32):+.3f} | {np.mean(slopes):.2f} |" if u32 else f'| {k} | {64 * k} | 0 | ' + ' | '.join(cells) + ' | - | - |')


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
