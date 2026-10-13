"""Experiment-1 table (mean +/- SE over seeds) for a stage: e1 (no replay) or e1r (replay)."""
import json
import os
import sys

import numpy as np

R = '/iris/u/lucyshi/fpl_transfer/runs'
stage = sys.argv[1]
seeds = [0, 1, 2]


def res(p):
    return json.load(open(p))['results'] if os.path.exists(p) else None


def fmt(v):
    v = [x for x in v if x is not None]
    if not v:
        return 'n/a'
    return f'{np.mean(v):.2f} ± {np.std(v, ddof=1) / np.sqrt(len(v)):.2f}' if len(v) > 1 else f'{v[0]:.2f}'


def get(r, peg):
    return r[peg]['success'] if r else None


base = {s: res(f'{R}/e1/k20_s{s}/arollouts/base_eval.json') for s in seeds}
print(f'| arm | A | B | ΔA | ΔB | B after adapting |')
print('|---|---|---|---|---|---|')
ad = [get(res(f'{R}/e1/k20_s{s}/base/base_adaptB_eval.json'), 'right') for s in seeds]
print(f"| base policy | {fmt([get(base[s], 'left') for s in seeds])} | {fmt([get(base[s], 'right') for s in seeds])} | | | {fmt(ad)} |")
for m in ['sft', 'filtered_bc', 'awr', 'udrl_scalar', 'udrl_axes']:
    for n in (200, 1000):
        ev = {s: res(f'{R}/{stage}/k20_s{s}/{m}/n{n}_eval.json') for s in seeds}
        if not any(ev.values()):
            continue
        A = [get(ev[s], 'left') for s in seeds]
        B = [get(ev[s], 'right') for s in seeds]
        dA = [a - get(base[s], 'left') if a is not None else None for s, a in zip(seeds, A)]
        dB = [b - get(base[s], 'right') if b is not None else None for s, b in zip(seeds, B)]
        AB = [get(res(f'{R}/{stage}/k20_s{s}/{m}/n{n}_adaptB_eval.json'), 'right') for s in seeds]
        print(f'| {m}, N={n} | {fmt(A)} | {fmt(B)} | {fmt(dA)} | {fmt(dB)} | {fmt(AB)} |')
ce = [res(f'{R}/e1/k20_s{s}/base/ceilB_eval.json') for s in seeds]
print(f"| ceiling: 1000 B demos (no replay) | {fmt([get(c, 'left') for c in ce])} | {fmt([get(c, 'right') for c in ce])} | | | |")
