"""
E1 progress: per run, iterations done and the evaluated budgets; then the per-K mean changes at each budget
over the runs evaluated there so far (training tasks, H, rest of R, U).

  python -m ktask.status
"""

import glob
import json
import os
import re

import numpy as np

from ktask.analyze import RUNS, base_eval, e1_runs, load_eval
from ktask.data import SPLITS


def main():
    rows = e1_runs()
    budgets = (4096, 8192, 16384, 32768)
    started = done = 0
    per = {}
    for s, c, k, d, train in rows:
        log = f'{d}/ppo_log.txt'
        its = sum(1 for l in open(log) if ' iter ' in l) if os.path.exists(log) else 0
        evals = [b for b in budgets if os.path.exists(f'{d}/eval_ep{b}.json')]
        started += its > 0
        done += 32768 in evals
        b0 = base_eval(s)
        for b in evals:
            e = load_eval(f'{d}/eval_ep{b}.json')
            rest = [t for t in SPLITS['R'] if t not in train]
            g = {'train': train, 'H': SPLITS['H'], 'rest': rest, 'U': SPLITS['U']}
            per.setdefault((k, b), []).append({n: np.mean([e[t]['success'].mean() - b0[t]['success'].mean()
                                                           for t in ts]) if ts else np.nan for n, ts in g.items()})
    print(f'E1: {len(rows)} runs, {started} started, {done} evaluated at 32768 episodes')
    print('mean change vs base (runs): K | budget | train | H | rest of R | U')
    for (k, b) in sorted(per):
        v = per[(k, b)]
        m = {n: np.nanmean([r[n] for r in v]) for n in ('train', 'H', 'rest', 'U')}
        print(f"K={k:2d} {b:6d}  n={len(v):2d} | train {m['train']:+.3f} | H {m['H']:+.3f} | "
              f"rest {m['rest']:+.3f} | U {m['U']:+.3f}")


if __name__ == '__main__':
    main()
