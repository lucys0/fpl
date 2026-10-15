"""E1 transfer per unit of training change and final changes, split by base seed.

  python -m ktask.by_seed
"""
import numpy as np

from ktask.analyze import base_eval, e1_runs, load_eval
from ktask.data import SPLITS
from ktask.plot_e1 import mean_delta, train_delta

BUDGETS = [4096, 8192, 16384, 32768]


def main():
    res = {}
    for s, c, k, d, train in e1_runs():
        b = base_eval(s)
        pts = []
        for n in BUDGETS:
            e = load_eval(f'{d}/eval_ep{n}.json')
            if e is None:
                break
            pts.append((train_delta(s, c, k, d, train, n, e, b), mean_delta(e, b, SPLITS['H']),
                        mean_delta(e, b, SPLITS['U'])))
        if len(pts) < len(BUDGETS):
            continue
        x = np.array([p[0] for p in pts])
        y = np.array([p[1] for p in pts])
        res.setdefault((s, k), []).append(((x * y).sum() / (x * x).sum(), *pts[-1]))
    print('seed  K runs | H per unit (per run)                  | final: training   H      U')
    for s in (0, 1, 2):
        for k in (1, 2, 4, 8, 16):
            r = np.array(res.get((s, k), []))
            if not len(r):
                continue
            runs = ' '.join(f'{v:+.2f}' for v in r[:, 0])
            print(f's{s} {k:3d} {len(r):4d} | {r[:, 0].mean():+.2f} ({runs}) | {r[:, 1].mean():+.3f} '
                  f'{r[:, 2].mean():+.3f} {r[:, 3].mean():+.3f}')


if __name__ == '__main__':
    main()
