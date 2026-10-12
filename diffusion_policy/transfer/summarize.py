"""
Summaries of the transfer experiments (mean +/- SE over seeds).

  python -m transfer.summarize e0                 # base policies vs demos per peg
  python -m transfer.summarize e1 --base_k 20     # post-training on A: A, B, adaptation to B

Task A = left peg, task B = right peg. Changes (delta) are measured against
each seed's own base policy, on the same held-out nut positions.
"""

import glob
import json
import os
import re

import click
import numpy as np

RUNS = '/iris/u/lucyshi/fpl_transfer/runs'
OUT = '/iris/u/lucyshi/fpl_transfer/results'


def load(path):
    return json.load(open(path))['results'] if os.path.exists(path) else None


def ms(values, digits=2):
    v = [x for x in values if x is not None]
    if not v:
        return 'n/a'
    se = np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return f'{np.mean(v):.{digits}f} ± {se:.{digits}f}' + ('' if len(v) == len(values) else f' (n={len(v)})')


@click.group()
def cli():
    pass


@cli.command()
def e0():
    rows = {}
    for path in sorted(glob.glob(f'{RUNS}/e0/base_k*_s*/eval.json')):
        k, s = map(int, re.search(r'base_k(\d+)_s(\d+)', path).groups())
        rows.setdefault(k, []).append(load(path))
    lines = ['| demos per peg | seeds | success A (left) | success B (right) | wrong peg A | wrong peg B | T1S A | T1S B |',
             '|---|---|---|---|---|---|---|---|']
    for k in sorted(rows):
        r = rows[k]
        lines.append(f"| {k} | {len(r)} | {ms([x['left']['success'] for x in r])} | {ms([x['right']['success'] for x in r])} | "
                     f"{ms([x['left']['wrong_peg'] for x in r])} | {ms([x['right']['wrong_peg'] for x in r])} | "
                     f"{ms([x['left']['t1s'] for x in r], 0)} | {ms([x['right']['t1s'] for x in r], 0)} |")
    wide = {}
    for path in sorted(glob.glob(f'{RUNS}/e0w/k*/eval_s*.json')):
        k = int(re.search(r'e0w/k(\d+)/', path).group(1))
        wide.setdefault(k, []).append(load(path))
    if wide:
        lines += ['', 'Same policies on the wide nut range:', '',
                  '| demos per peg | seeds | success A (left) | success B (right) |', '|---|---|---|---|']
        for k in sorted(wide):
            r = wide[k]
            lines.append(f"| {k} | {len(r)} | {ms([x['left']['success'] for x in r])} | {ms([x['right']['success'] for x in r])} |")
    print('\n'.join(lines))


@cli.command()
@click.option('--base_k', required=True, type=int)
def e1(base_k):
    os.makedirs(OUT, exist_ok=True)
    seeds = sorted(int(re.search(r'_s(\d+)$', d).group(1)) for d in glob.glob(f'{RUNS}/e1/k{base_k}_s*'))
    base = {s: load(f'{RUNS}/e1/k{base_k}_s{s}/arollouts/base_eval.json') for s in seeds}   # wide range
    base_adapt = {s: load(f'{RUNS}/e1/k{base_k}_s{s}/base/base_adaptB_eval.json') for s in seeds}
    ceil = {s: load(f'{RUNS}/e1/k{base_k}_s{s}/base/ceilB_eval.json') for s in seeds}

    def col(fn):
        return [fn(s) for s in seeds]

    lines = [f'# Transfer from task A (left peg) to task B (right peg), base policies on {base_k} demos per peg', '',
             f'Seeds: {seeds}. Success on the commanded peg, {100} held-out nut positions per peg per seed.', '',
             '| arm | A success | B success | ΔA | ΔB | ΔB / ΔA | B after adapting (200 B rollouts, filtered BC) |',
             '|---|---|---|---|---|---|---|']
    sa = lambda r, peg: r[peg]['success'] if r else None
    lines.append(f"| base policy | {ms(col(lambda s: sa(base[s], 'left')))} | {ms(col(lambda s: sa(base[s], 'right')))} "
                 f"| — | — | — | {ms(col(lambda s: sa(base_adapt[s], 'right')))} |")
    points = []
    methods = ['sft', 'filtered_bc', 'awr', 'udrl_scalar', 'udrl_axes']
    for m in methods:
        for n in (200, 1000):
            ev = {s: load(f'{RUNS}/e1/k{base_k}_s{s}/{m}/n{n}_eval.json') for s in seeds}
            ad = {s: load(f'{RUNS}/e1/k{base_k}_s{s}/{m}/n{n}_adaptB_eval.json') for s in seeds}
            if not any(ev.values()):
                continue
            dA = col(lambda s: sa(ev[s], 'left') - sa(base[s], 'left') if ev[s] and base[s] else None)
            dB = col(lambda s: sa(ev[s], 'right') - sa(base[s], 'right') if ev[s] and base[s] else None)
            ratio = col(lambda s: (sa(ev[s], 'right') - sa(base[s], 'right')) / (sa(ev[s], 'left') - sa(base[s], 'left'))
                        if ev[s] and base[s] and abs(sa(ev[s], 'left') - sa(base[s], 'left')) >= 0.05 else None)
            lines.append(f"| {m}, N={n} | {ms(col(lambda s: sa(ev[s], 'left')))} | {ms(col(lambda s: sa(ev[s], 'right')))} "
                         f"| {ms(dA)} | {ms(dB)} | {ms(ratio)} | {ms(col(lambda s: sa(ad[s], 'right')))} |")
            points += [(m, n, a, b) for a, b in zip(dA, dB) if a is not None and b is not None]
    lines.append(f"| ceiling: {1000} B demos | {ms(col(lambda s: sa(ceil[s], 'left')))} | {ms(col(lambda s: sa(ceil[s], 'right')))} | | | | |")
    lines += ['', 'ΔB/ΔA is computed per seed and only where |ΔA| ≥ 0.05.']
    text = '\n'.join(lines)
    print(text)
    open(f'{OUT}/e1_k{base_k}.md', 'w').write(text + '\n')

    if points:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 5))
        colors = dict(zip(methods, ['tab:gray', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red']))
        for m in methods:
            for n, marker in ((200, 'o'), (1000, 's')):
                p = [(a, b) for mm, nn, a, b in points if mm == m and nn == n]
                if p:
                    ax.scatter(*zip(*p), c=colors[m], marker=marker, label=f'{m} N={n}', alpha=0.8)
        ax.axhline(0, c='k', lw=0.5); ax.axvline(0, c='k', lw=0.5)
        ax.set_xlabel('Δ success on A (trained)'); ax.set_ylabel('Δ success on B (not trained)')
        ax.set_title('Post-training on A: gain on A vs change on B (one point per seed)')
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(f'{OUT}/e1_k{base_k}_dA_dB.png', dpi=150)
        print(f'figure: {OUT}/e1_k{base_k}_dA_dB.png')


if __name__ == '__main__':
    cli()
