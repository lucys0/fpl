"""
Paired-bootstrap confidence intervals for the redesigned transfer experiment.

Every arm and its seed's base policy are evaluated from the same 100 start states
per peg, so changes are computed per start state and resampled together. One
bootstrap sample redraws the start states within each seed (seeds stay fixed,
there are only three); a change is the mean over seeds and start states.

Reports, per arm: ΔA, ΔB (95% CI), and the transfer ratio ΔB / ΔA (CI from the
same samples; undefined when ΔA's CI includes 0).

  python -m transfer.m_stats [--seeds 0,1,2] [--n_boot 5000]
"""

import json
import os

import click
import numpy as np

R = '/iris/u/lucyshi/fpl_transfer/runs'


def episodes(p, peg):
    if not os.path.exists(p):
        return None
    eps = json.load(open(p))['results'][peg]['episodes']
    return {e['seed']: float(e['success']) for e in eps}


def arm_paths(stage, arm, n, s):
    if stage == 'm2':
        return f'{R}/m2/s{s}/ppo/n{n}_eval.json'
    if stage == 'm5':
        return f'{R}/m5/s{s}/{arm}/step{n}_eval.json'
    return f'{R}/{stage}/s{s}/{arm}/n{n}_eval.json'


ARMS = ([('m1r', m, n) for m in ('sft_pool', 'filtered_bc', 'awr', 'udrl_scalar', 'udrl_axes', 'sft_expert')
         for n in (200, 1000)]
        + [('m1', m, n) for m in ('sft_pool', 'filtered_bc', 'awr', 'udrl_scalar', 'udrl_axes', 'sft_expert')
           for n in (200, 1000)]
        + [('m2', 'ppo', n) for n in (200, 1000, 3000, 6000)]
        + [('m4', m, n) for m in ('filtered_bc', 'awr') for n in (200, 1000, 3000)]
        + [('m4', 'filtered_bc_lr1e-06', n) for n in (200, 400, 600, 800, 1000)]
        + [('m5', f'{m}_lr{lr}', st) for m in ('filtered_bc', 'sft_expert')
           for lr, sts in (('1e-04', (100, 200, 500, 1000)), ('1e-05', (500, 1000, 2500, 5000)),
                           ('1e-06early', (100, 200, 400, 700, 1000)), ('1e-06', (1000, 3000, 6000, 10000)))
           for st in sts])


@click.command()
@click.option('--seeds', default='0,1,2')
@click.option('--n_boot', default=5000)
@click.option('--min_seeds', default=3)
def main(seeds, n_boot, min_seeds):
    seeds = [int(s) for s in seeds.split(',')]
    rng = np.random.default_rng(0)
    print('| stage | arm | budget | seeds | ΔA [95% CI] | ΔB [95% CI] | ΔB / ΔA [95% CI] |')
    print('|---|---|---|---|---|---|---|')
    for stage, arm, n in ARMS:
        diffs = {}                                  # peg -> list over seeds of per-start differences
        for peg in ('left', 'right'):
            per_seed = []
            for s in seeds:
                b = episodes(f'{R}/m0/base_s{s}/eval.json', peg)
                a = episodes(arm_paths(stage, arm, n, s), peg)
                if a is None or b is None:
                    continue
                common = sorted(set(a) & set(b))
                per_seed.append(np.array([a[k] - b[k] for k in common]))
            diffs[peg] = per_seed
        k = min(len(diffs['left']), len(diffs['right']))
        if k < min_seeds:
            continue
        boot = {peg: np.zeros(n_boot) for peg in diffs}
        for peg, per_seed in diffs.items():
            for d in per_seed:
                idx = rng.integers(0, len(d), size=(n_boot, len(d)))
                boot[peg] += d[idx].mean(axis=1) / len(per_seed)
        point = {peg: np.mean([d.mean() for d in diffs[peg]]) for peg in diffs}
        ci = {peg: np.percentile(boot[peg], [2.5, 97.5]) for peg in diffs}
        cell = lambda peg: f"{point[peg]:+.2f} [{ci[peg][0]:+.2f}, {ci[peg][1]:+.2f}]"
        if ci['left'][0] > 0:
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = boot['right'] / boot['left']
            lo, hi = np.percentile(ratio, [2.5, 97.5])
            rcell = f"{point['right'] / point['left']:.2f} [{lo:.2f}, {hi:.2f}]"
        else:
            rcell = 'ΔA not > 0'
        print(f"| {stage} | {arm} | {n} | {k} | {cell('left')} | {cell('right')} | {rcell} |")


if __name__ == '__main__':
    main()
