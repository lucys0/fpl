"""
Table for the redesigned transfer experiment (mean +/- SE over seeds): task A
(left peg, post-trained) for context, task B (right peg, never post-trained)
zero-shot with its grasp / placement split, and B after learning from 10 clean
B demos. Changes are against each seed's base policy on the same starts.

  python -m transfer.m_table [--seeds 0,1,2]
"""

import json
import os

import click
import numpy as np

R = '/iris/u/lucyshi/fpl_transfer/runs'
METHODS = ('sft_pool', 'filtered_bc', 'awr', 'udrl_scalar', 'udrl_axes', 'sft_expert')


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
    return r[peg][key] if r and r[peg].get(key) is not None else None


@click.command()
@click.option('--seeds', default='0,1,2')
def main(seeds):
    seeds = [int(s) for s in seeds.split(',')]
    base = {s: res(f'{R}/m0/base_s{s}/eval.json') for s in seeds}
    base_learn = {s: res(f'{R}/m0/base_s{s}/base_learnB_eval.json') for s in seeds}
    head = ('| arm | A | B | ΔA | ΔB | B grasp | B place given grasp | B after 10 B demos | Δ vs base after 10 B demos |\n'
            '|---|---|---|---|---|---|---|---|---|')

    def row(name, ev, learn):
        dA = [g(ev[s], 'left') - g(base[s], 'left') if ev[s] and base[s] else None for s in seeds]
        dB = [g(ev[s], 'right') - g(base[s], 'right') if ev[s] and base[s] else None for s in seeds]
        dL = [g(learn[s], 'right') - g(base_learn[s], 'right') if learn[s] and base_learn[s] else None for s in seeds]
        return (f"| {name} | {ms([g(ev[s], 'left') for s in seeds])} | {ms([g(ev[s], 'right') for s in seeds])} | "
                f"{ms(dA)} | {ms(dB)} | {ms([g(ev[s], 'right', 'grasp') for s in seeds])} | "
                f"{ms([g(ev[s], 'right', 'place_given_grasp') for s in seeds])} | "
                f"{ms([g(learn[s], 'right') for s in seeds])} | {ms(dL)} |")

    lines = [head, row('base policy', base, base_learn).replace('| n/a | n/a |', '| | |', 1)]
    out = {}
    for stage, title in (('m1r', 'with replay of the pretraining pool'), ('m1', 'without replay')):
        block = []
        for m in METHODS:
            for n in (200, 1000):
                ev = {s: res(f'{R}/{stage}/s{s}/{m}/n{n}_eval.json') for s in seeds}
                if any(ev.values()):
                    learn = {s: res(f'{R}/{stage}/s{s}/{m}/n{n}_learnB_eval.json') for s in seeds}
                    block.append(row(f'{m}, N={n}', ev, learn))
        out[title] = block
    ppo = []
    for n in (200, 1000, 3000, 6000):
        ev = {s: res(f'{R}/m2/s{s}/ppo/n{n}_eval.json') for s in seeds}
        if any(ev.values()):
            learn = {s: res(f'{R}/m2/s{s}/ppo/n{n}_learnB_eval.json') for s in seeds}
            ppo.append(row(f'ppo, {n} episodes', ev, learn))
    text = '\n'.join(lines)
    for title, block in out.items():
        if block:
            text += f'\n\n**Post-training on A, {title}**\n\n' + head + '\n' + '\n'.join(block)
    if ppo:
        text += '\n\n**PPO on A (online, no replay)**\n\n' + head + '\n' + '\n'.join(ppo)
    print(text)


if __name__ == '__main__':
    main()
