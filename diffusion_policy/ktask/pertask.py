"""Per-task changes of one evaluation vs its base, grouped by split, object, and target.

  python -m ktask.pertask EVAL_JSON --seed S
"""
import sys

import click
import numpy as np

from ktask.analyze import base_eval, load_eval
from ktask.data import SPLITS
from ktask.env import OBJ_NAMES, TGT_NAMES, task_name


@click.command()
@click.argument('path')
@click.option('--seed', type=int, required=True)
def main(path, seed):
    b, e = base_eval(seed), load_eval(path)
    d = {t: e[t]['success'].mean() - b[t]['success'].mean() for t in e}
    for split in ('R', 'H', 'U'):
        rows = sorted(SPLITS[split], key=lambda t: -d[t])
        print(f'{split}: ' + ', '.join(f"{task_name(t)} {b[t]['success'].mean():.2f}->{e[t]['success'].mean():.2f}"
                                       for t in rows))
    print('by object: ' + ', '.join(f'{o} {np.mean([d[t] for t in d if t // 4 == i]):+.2f}' for i, o in enumerate(OBJ_NAMES)))
    print('by target: ' + ', '.join(f'{g} {np.mean([d[t] for t in d if t % 4 == i]):+.2f}' for i, g in enumerate(TGT_NAMES)))


if __name__ == '__main__':
    main()
