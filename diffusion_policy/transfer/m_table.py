"""
Table for the redesigned transfer experiment (mean +/- SE over seeds): task A
(left peg, post-trained) for context, task B (right peg, never post-trained)
zero-shot with its grasp / placement split, and B after learning from K clean
B demos. Changes are against each seed's base policy on the same starts.

  python -m transfer.m_table [--seeds 0,1,2] [--ks 10,3]

Learnability results live next to the checkpoint's job (<tag>_learnB[_k<K>]_eval.json)
or, when added later by the m3 stage, under runs/m3/<stage>/s<seed>/<arm>/.
"""

import json
import os

import click
import numpy as np

R = '/iris/u/lucyshi/fpl_transfer/runs'
METHODS = ('sft_pool', 'filtered_bc', 'awr', 'udrl_scalar', 'udrl_axes', 'sft_expert')
LEARN_B_K = 10


def res(p):
    return json.load(open(p))['results'] if os.path.exists(p) else None


def ms(v, d=2):
    v = [x for x in v if x is not None]
    if not v:
        return 'n/a'
    if len(v) == 1:
        return f'{v[0]:.{d}f} (n=1)'
    return f'{np.mean(v):.{d}f} ± {np.std(v, ddof=1) / np.sqrt(len(v)):.{d}f}'


def g(r, peg, key='success'):
    return r[peg][key] if r and r.get(peg) and r[peg].get(key) is not None else None


def learn(job_dir, m3_dir, tag, k):
    name = f'{tag}_learnB' + ('' if k == LEARN_B_K else f'_k{k}') + '_eval.json'
    return res(f'{job_dir}/{name}') or res(f'{m3_dir}/{name}')


@click.command()
@click.option('--seeds', default='0,1,2')
@click.option('--ks', default='10,3', help='Numbers of clean B demos in the learnability columns')
def main(seeds, ks):
    seeds = [int(s) for s in seeds.split(',')]
    ks = [int(k) for k in ks.split(',')]
    base = {s: res(f'{R}/m0/base_s{s}/eval.json') for s in seeds}
    base_learn = {k: {s: learn(f'{R}/m0/base_s{s}', f'{R}/m3/m0/s{s}/base', 'base', k) for s in seeds} for k in ks}
    head = ('| arm | A | B | ΔA | ΔB | B grasp | B place given grasp | '
            + ' | '.join(f'B after {k} B demos | Δ vs base' for k in ks) + ' |\n'
            + '|---' * (7 + 2 * len(ks)) + '|')

    def row(name, ev, lr):
        d = lambda a, b, peg: [g(a[s], peg) - g(b[s], peg) if g(a[s], peg) is not None and g(b[s], peg) is not None
                               else None for s in seeds]
        cells = [ms([g(ev[s], 'left') for s in seeds]), ms([g(ev[s], 'right') for s in seeds]),
                 ms(d(ev, base, 'left')), ms(d(ev, base, 'right')),
                 ms([g(ev[s], 'right', 'grasp') for s in seeds]),
                 ms([g(ev[s], 'right', 'place_given_grasp') for s in seeds])]
        for k in ks:
            cells += [ms([g(lr[k][s], 'right') for s in seeds]), ms(d(lr[k], base_learn[k], 'right'))]
        return f"| {name} | " + ' | '.join(cells) + ' |'

    text = head + '\n' + row('base policy', base, base_learn)
    for stage, title in (('m1r', 'with replay of the pretraining pool'), ('m1', 'without replay')):
        block = []
        for m in METHODS:
            for n in (200, 1000):
                ev = {s: res(f'{R}/{stage}/s{s}/{m}/n{n}_eval.json') for s in seeds}
                if any(ev.values()):
                    lr = {k: {s: learn(f'{R}/{stage}/s{s}/{m}', f'{R}/m3/{stage}/s{s}/{m}', f'n{n}', k) for s in seeds}
                          for k in ks}
                    block.append(row(f'{m}, N={n}', ev, lr))
        if block:
            text += f'\n\n**Post-training on A, {title}**\n\n' + head + '\n' + '\n'.join(block)
    ppo = []
    for n in (200, 1000, 3000, 6000):
        ev = {s: res(f'{R}/m2/s{s}/ppo/n{n}_eval.json') for s in seeds}
        if any(ev.values()):
            lr = {k: {s: learn(f'{R}/m2/s{s}/ppo', f'{R}/m3/m2/s{s}/ppo', f'n{n}', k) for s in seeds} for k in ks}
            ppo.append(row(f'ppo, {n} episodes', ev, lr))
    for m in ('filtered_bc', 'awr'):
        for n in (200, 1000, 3000):
            ev = {s: res(f'{R}/m4/s{s}/{m}/n{n}_eval.json') for s in seeds}
            if any(ev.values()):
                lr = {k: {s: learn(f'{R}/m4/s{s}/{m}', f'{R}/m3/m4/s{s}/{m}', f'n{n}', k) for s in seeds} for k in ks}
                ppo.append(row(f'iterated {m}, {n} episodes', ev, lr))
    if ppo:
        text += ('\n\n**Online on A, no replay (PPO, and filtered BC / AWR iterated on their own rollouts)**\n\n'
                 + head + '\n' + '\n'.join(ppo))
    print(text)


if __name__ == '__main__':
    main()
