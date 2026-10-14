"""
Tables for the third round of experiments (2026-10-13), from whatever evaluations exist so far.
Differences are paired by start state against a reference on the same seed; 95% CIs bootstrap start
states within seeds (seeds fixed). n = seeds with results.

  python -m transfer.round3 [--sections a,xlearn,logged,time,xppob,probes,xpeg]
"""

import json
import os

import click
import numpy as np

R = '/iris/u/lucyshi/fpl_transfer/runs'
SEEDS = (0, 1, 2)
RNG = np.random.default_rng(0)
PPO_GAMMA_STEP = 0.99 ** (1 / 8)


def episodes(path, task):
    """{start seed: episode} for one evaluation file and task, or None."""
    if not path or not os.path.exists(path):
        return None
    r = json.load(open(path))['results']
    return {e['seed']: e for e in r[task]['episodes']} if task in r else None


def stat(paths, task, ref_paths=None, metric='success'):
    """Mean over seeds (and, with ref_paths, the paired difference and its CI)."""
    per = []
    for s in SEEDS:
        a = episodes(paths(s), task)
        b = episodes(ref_paths(s), task) if ref_paths else None
        if a is None or (ref_paths and b is None):
            continue
        keys = sorted(a)
        get = metric if callable(metric) else (lambda e: e[metric])
        va = np.array([float(get(a[k])) for k in keys])
        vb = np.array([float(get(b[k])) for k in keys]) if b else None
        per.append((va, vb))
    if not per:
        return None
    mean = np.mean([va.mean() for va, _ in per])
    if not ref_paths:
        return {'n': len(per), 'mean': mean}
    d = np.mean([(va - vb).mean() for va, vb in per])
    boots = []
    for _ in range(2000):
        bs = []
        for va, vb in per:
            i = RNG.integers(0, len(va), len(va))
            bs.append((va[i] - vb[i]).mean())
        boots.append(np.mean(bs))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {'n': len(per), 'mean': mean, 'd': d, 'lo': lo, 'hi': hi}


def fmt(r, delta=True):
    if r is None:
        return '-'
    if delta and 'd' in r:
        return f"{r['mean']:.2f} ({r['d']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]) n={r['n']}"
    return f"{r['mean']:.2f} n={r['n']}"


m_base = lambda s: f'{R}/m0/base_s{s}/eval.json'
x_base = lambda s: f'{R}/x0/base_s{s}/eval.json'


def table(title, header, rows):
    print(f'\n### {title}\n')
    print('| ' + ' | '.join(header) + ' |')
    print('|' + '---|' * len(header))
    for row in rows:
        print('| ' + ' | '.join(row) + ' |')


# ---------------------------------------------------------------- A-only bases (B never in pretraining)

def section_a():
    a0 = lambda s: f'{R}/a0/base_s{s}/eval/eval.json'
    starts = {'base (BC on A)': a0,
              'BC on A, then PPO on A (6000 ep.)': lambda s: f'{R}/a1/s{s}/ppo/eval_n6000/eval.json',
              'BC on A, then SFT on 1000 clean A demos': lambda s: f'{R}/a1/s{s}/sft_expert/eval/eval.json',
              'BC on A, then filtered BC on the A pool': lambda s: f'{R}/a1/s{s}/filtered_bc/eval/eval.json'}
    rows = []
    for name, p in starts.items():
        ref = None if p is a0 else a0
        on_a_peg = lambda e: e['final_peg'] == 'left'
        rows.append([name] + [fmt(stat(p, t, ref)) for t in ('left', 'right', 'round')]
                    + [fmt(stat(p, t, None, on_a_peg), False) for t in ('right', 'round')])
    table('B never in pretraining: A, and B zero-shot (an unseen goal input); changes vs the A-only base',
          ['start', 'A (square nut, left peg)', 'B1 zero-shot (right peg)', 'B2 zero-shot (round nut)',
           'B1: nut put on A\'s peg', 'B2: nut put on A\'s peg'], rows)
    for n in (1000, 3000):
        p = lambda s, n=n: f'{R}/a1/s{s}/ppo/eval_n{n}/eval.json'
        r = stat(p, 'left', a0)
        if r:
            print(f'PPO on A, {n} ep.: A {fmt(r)}')
    keys = {'base': 'base', 'ppoA6000': 'PPO on A', 'sft_expert': 'SFT on A', 'filtered_bc': 'filtered BC on A'}
    for task, tname in (('right', 'B1 = right peg'), ('round', 'B2 = round nut')):
        rows = []
        ks = (5, 10, 25, 50, 100)
        for key, name in keys.items():
            ref = None if key == 'base' else (lambda s, k: f'{R}/a2/s{s}/base/{task}/eval_k{k}/eval.json')
            row = [name]
            for k in ks:
                p = lambda s, k=k: f'{R}/a2/s{s}/{key}/{task}/eval_k{k}/eval.json'
                row.append(fmt(stat(p, task, (lambda s, k=k: ref(s, k)) if ref else None)))
            rows.append(row)
        table(f'Learning {tname} from k clean demos (SFT, 2000 steps), from starts that never saw B; '
              f'vs the A-only base at the same k', ['start'] + [f'k={k}' for k in ks], rows)


def section_a3(k=5):
    """PPO on B from each A-only start, after SFT on k clean B demos."""
    for task, tname in (('right', 'B1 = right peg'), ('round', 'B2 = round nut')):
        rows = []
        for key, name in (('base', 'A-only base'), ('ppoA6000', 'PPO on A'), ('sft_expert', 'SFT on A')):
            warm = lambda s, key=key: f'{R}/a2/s{s}/{key}/{task}/eval_k{k}/eval.json'
            warm_ref = lambda s: f'{R}/a2/s{s}/base/{task}/eval_k{k}/eval.json'
            row = [name, fmt(stat(warm, task, None if key == 'base' else warm_ref))]
            for n in (200, 1000, 3000):
                p = lambda s, n=n, key=key: f'{R}/a3/s{s}/{key}/{task}_k{k}/eval_n{n}/eval.json'
                ref = None if key == 'base' else (lambda s, n=n: f'{R}/a3/s{s}/base/{task}_k{k}/eval_n{n}/eval.json')
                row.append(fmt(stat(p, task, ref)))
            p = lambda s, key=key: f'{R}/a3/s{s}/{key}/{task}_k{k}/eval_n3000/eval.json'
            row.append(fmt(stat(p, 'left'), False))
            rows.append(row)
        table(f'B never in pretraining: PPO on {tname} after SFT on {k} clean demos (vs from the A-only base)',
              ['start', f'B after {k} demos', 'B after 200 PPO ep.', '1000', '3000', 'A after 3000'], rows)


# ---------------------------------------------------------------- second pair: learning the round nut

def section_xlearn():
    starts = {'base': 'base', 'ppoA6000': 'PPO on A (6000 ep.)', 'sft_expert_r1000': 'SFT expert, replay',
              'filtered_bc_r1000': 'filtered BC, replay', 'sft_expert_lr1e6_400': 'SFT expert, lr 1e-6, 400 steps'}
    rows = []
    for key, name in starts.items():
        row = [name]
        for k in (3, 5, 10):
            p = lambda s, k=k: f'{R}/xlearn/s{s}/{key}/eval_k{k}/eval.json'
            ref = None if key == 'base' else (lambda s, k=k: f'{R}/xlearn/s{s}/base/eval_k{k}/eval.json')
            row.append(fmt(stat(p, 'round', ref)))
        zero = {'base': x_base, 'ppoA6000': lambda s: f'{R}/x2/s{s}/ppo/n6000_eval.json',
                'sft_expert_r1000': lambda s: f'{R}/x1r/s{s}/sft_expert/n1000_eval.json',
                'filtered_bc_r1000': lambda s: f'{R}/x1r/s{s}/filtered_bc/n1000_eval.json',
                'sft_expert_lr1e6_400': lambda s: f'{R}/x5/s{s}/sft_expert_lr1e-06early/step400_eval.json'}[key]
        rows.append([row[0], fmt(stat(zero, 'round'), False)] + row[1:])
    table('Second pair: round-nut success after SFT on k clean round-nut demos (vs the base at the same k)',
          ['start', 'k=0 (zero-shot)', 'k=3', 'k=5', 'k=10'], rows)


# ---------------------------------------------------------------- same data, different update rule

def section_logged(stages=('mlog', 'xlog')):
    for stage in stages:
        pair1 = stage.startswith('m')
        base, B = (m_base, 'right') if pair1 else (x_base, 'round')
        rows = []
        for n in (1000, 3000, 6000):
            p = lambda s, n=n: f'{R}/{stage}/s{s}/ppo/eval_n{n}/eval.json'
            rows.append([f'PPO, {n} ep.', fmt(stat(p, 'left', base)), fmt(stat(p, B, base))])
        arms = (('filtered_bc', 'filtered BC on PPO\'s episodes'), ('awr', 'AWR on PPO\'s episodes')) \
            if stage.endswith('log') else (('fast_bc', 'BC on the fastest 30% of PPO\'s successes'),
                                           ('awr_time', 'AWR with the time cost on PPO\'s episodes'))
        for arm, name in arms:
            for st in (1000, 3000, 6000, 10000):
                p = lambda s, st=st: f'{R}/{stage}/s{s}/{arm}/eval_step{st}/eval.json'
                rows.append([f'{name}, lr 1e-6, {st} steps', fmt(stat(p, 'left', base)), fmt(stat(p, B, base))])
        table(f"{stage}: {'first' if pair1 else 'second'} pair, PPO (episodes logged"
              f"{', time cost 2' if 'time' in stage else ''}) and offline learners on exactly its episodes; "
              f"vs base", ['arm', 'A', f'B ({B})'], rows)


def time_rows(stage):
    """Time to success on start states where base and arm both succeed (relative change), A and B."""
    pair1 = stage.startswith('m')
    base, B = (m_base, 'right') if pair1 else (x_base, 'round')
    arms = [(f'PPO, {n} ep.', lambda s, n=n: f'{R}/{stage}/s{s}/ppo/eval_n{n}/eval.json') for n in (1000, 3000, 6000)]
    if stage.endswith('time'):
        arms += [(f'{nm}, {st} steps', lambda s, st=st, a=a: f'{R}/{stage}/s{s}/{a}/eval_step{st}/eval.json')
                 for a, nm in (('fast_bc', 'fastest-30% BC'), ('awr_time', 'time-cost AWR')) for st in (3000, 10000)]
    rows = []
    for name, p in arms:
        row = [name]
        for task in ('left', B):
            pairs = []
            for s in SEEDS:
                a, b = episodes(p(s), task), episodes(base(s), task)
                if a is None or b is None:
                    continue
                pairs += [(b[k]['first_success_step'], a[k]['first_success_step']) for k in a
                          if a[k]['success'] and b[k]['success']]
            if len(pairs) < 5:
                row.append('-')
                continue
            x = np.array(pairs, dtype=float)
            boots = [(lambda y: y[:, 1].mean() / y[:, 0].mean() - 1)(x[RNG.integers(0, len(x), len(x))]) for _ in range(2000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            row.append(f'{x[:, 0].mean():.0f} → {x[:, 1].mean():.0f} ({x[:, 1].mean() / x[:, 0].mean() - 1:+.1%} '
                       f'[{lo:+.1%}, {hi:+.1%}]) pairs={len(x)}')
        rows.append(row)
    return rows, B


def section_time():
    for stage in ('mtime', 'xtime', 'mlog', 'xlog'):
        rows, B = time_rows(stage)
        table(f'{stage}: time to success (env steps) where base and arm both succeed',
              ['arm', 'A', f'B ({B})'], rows)
    section_logged(('mtime', 'xtime'))


# ---------------------------------------------------------------- PPO on the round nut

def section_xppob():
    starts = {'base': x_base, 'ppoA6000': lambda s: f'{R}/x2/s{s}/ppo/n6000_eval.json',
              'sft_expert_r1000': lambda s: f'{R}/x1r/s{s}/sft_expert/n1000_eval.json'}
    rows = []
    for key, zero in starts.items():
        row = [key, fmt(stat(zero, 'round'), False)]
        for n in (200, 1000, 3000):
            p = lambda s, n=n: f'{R}/xppob/s{s}/{key}/eval_n{n}/eval.json'
            ref = None if key == 'base' else (lambda s, n=n: f'{R}/xppob/s{s}/base/eval_n{n}/eval.json')
            row.append(fmt(stat(p, 'round', ref)))
        p = lambda s: f'{R}/xppob/s{s}/{key}/eval_n3000/eval.json'
        row.append(fmt(stat(p, 'left'), False))
        rows.append(row)
    table('PPO on the round nut from the second pair\'s checkpoints (vs from the base, same budget)',
          ['start', 'B at start', 'B after 200 ep.', '1000', '3000', 'A after 3000'], rows)


# ---------------------------------------------------------------- probes

def section_probes():
    probes = ('push4_50', 'push4_120', 'friction0.1')
    for pair, base, B, arms in (('m', m_base, 'right', ('base', 'ppoA6000', 'sft_expert_r1000', 'filtered_bc_r1000',
                                                        'sft_expert_lr1e6_700')),
                                ('x', x_base, 'round', ('base', 'ppoA6000', 'sft_expert_r1000', 'filtered_bc_r1000',
                                                        'sft_expert_lr1e6_400'))):
        for task in ('left', B):
            rows = []
            for arm in arms:
                row = [arm]
                for pr in probes:
                    p = lambda s, pr=pr: f'{R}/{pair}probe/s{s}/{arm}/{pr}/eval.json'
                    ref = None if arm == 'base' else (lambda s, pr=pr: f'{R}/{pair}probe/s{s}/base/{pr}/eval.json')
                    row.append(fmt(stat(p, task, ref)))
                rows.append(row)
            table(f"Probes, {'first' if pair == 'm' else 'second'} pair, task {task} (vs the base under the same probe)",
                  ['arm'] + list(probes), rows)


# ---------------------------------------------------------------- moved round peg

def section_xpeg():
    zero = {'base': x_base, 'ppoA6000': lambda s: f'{R}/x2/s{s}/ppo/n6000_eval.json',
            'sft_expert_r1000': lambda s: f'{R}/x1r/s{s}/sft_expert/n1000_eval.json'}
    ts = (-0.425, 0.0, 0.425, 0.85, 1.275, 1.7)
    rows = []
    for arm, z in zero.items():
        row = [arm]
        for t in ts:
            p = z if t == 0 else (lambda s, t=t: f'{R}/xpeg/s{s}/{arm}/t{t:+.3f}/eval.json')
            row.append(fmt(stat(p, 'round'), False))
        rows.append(row)
    table('Round peg moved along PPO\'s release shift (cm; 0.425 = the base-to-PPO shift): round-nut success',
          ['arm'] + [f'{t:+.3f}' for t in ts], rows)


@click.command()
@click.option('--sections', default='a,xlearn,logged,time,xppob,probes,xpeg')
def main(sections):
    fns = {'a': section_a, 'a3': section_a3, 'xlearn': section_xlearn, 'logged': section_logged, 'time': section_time,
           'xppob': section_xppob, 'probes': section_probes, 'xpeg': section_xpeg}
    for name in sections.split(','):
        fns[name]()


if __name__ == '__main__':
    main()
