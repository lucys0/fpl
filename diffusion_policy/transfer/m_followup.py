"""
Tables for the three follow-up experiments (mean ± SE over seeds, n = seeds with results; changes
against each seed's base policy on the same 100 start states per task).

  1. m7: BC held to PPO's KL from the base vs PPO at that KL
  2. m6: PPO on B from checkpoints post-trained on A (B and A success along B training)
  3. x*: the second task pair (B = the round nut, grasped by its ring, onto the round peg)

  python -m transfer.m_followup [--seeds 0,1,2]
"""

import json
import os

import click
import numpy as np

R = '/iris/u/lucyshi/fpl_transfer/runs'


def res(p):
    return json.load(open(p))['results'] if os.path.exists(p) else None


def succ(r, peg):
    return r[peg]['success'] if r and r.get(peg) else None


def ms(v):
    v = [x for x in v if x is not None]
    if not v:
        return 'n/a'
    if len(v) == 1:
        return f'{v[0]:.2f} (n=1)'
    return f'{np.mean(v):.2f} ± {np.std(v, ddof=1) / np.sqrt(len(v)):.2f} (n={len(v)})'


def dms(v):
    v = [x for x in v if x is not None]
    if not v:
        return 'n/a'
    if len(v) == 1:
        return f'{v[0]:+.2f} (n=1)'
    return f'{np.mean(v):+.2f} ± {np.std(v, ddof=1) / np.sqrt(len(v)):.2f} (n={len(v)})'


@click.command()
@click.option('--seeds', default='0,1,2')
def main(seeds):
    seeds = [int(s) for s in seeds.split(',')]
    base = {s: res(f'{R}/m0/base_s{s}/eval.json') for s in seeds}

    def delta(ev, b, peg):
        return [succ(ev[s], peg) - succ(b[s], peg) if succ(ev[s], peg) is not None and succ(b[s], peg) is not None
                else None for s in seeds]

    # ---------------- 1. KL-held BC
    kl = json.load(open('/iris/u/lucyshi/fpl_transfer/results/m_kl.json'))
    print('## 1. BC held to PPO\'s KL from the base (no replay; KL on A states)\n')
    print('| arm | KL budget (A states) | ΔA | ΔB |')
    print('|---|---|---|---|')
    for n in (3000, 6000):
        ppo = {s: res(f'{R}/m2/s{s}/ppo/n{n}_eval.json') for s in seeds}
        budget = np.mean([r['kl_A'] for r in kl[f'PPO | {n} ep.']])
        print(f'| PPO, {n} episodes | {budget:.1f} (its own) | {dms(delta(ppo, base, "left"))} | '
              f'{dms(delta(ppo, base, "right"))} |')
        for m in ('filtered_bc', 'sft_expert'):
            for st in (1000, 5000):
                ev = {s: res(f'{R}/m7/s{s}/{m}_klppo{n}/step{st}_eval.json') for s in seeds}
                if any(ev.values()):
                    print(f'| {m}, held to PPO-{n} KL, {st} steps | {budget:.1f} | {dms(delta(ev, base, "left"))} | '
                          f'{dms(delta(ev, base, "right"))} |')

    # ---------------- 2. PPO on B
    print('\n## 2. PPO on B from checkpoints post-trained on A\n')
    print('| start | B at start | B after 200 | 1000 | 3000 B episodes | A after 3000 B episodes |')
    print('|---|---|---|---|---|---|')
    starts = {'base': lambda s: f'{R}/m0/base_s{s}/eval.json',
              'ppoA6000': lambda s: f'{R}/m2/s{s}/ppo/n6000_eval.json',
              'filtered_bc_r1000': lambda s: f'{R}/m1r/s{s}/filtered_bc/n1000_eval.json',
              'sft_expert_r1000': lambda s: f'{R}/m1r/s{s}/sft_expert/n1000_eval.json',
              'awr_r1000': lambda s: f'{R}/m1r/s{s}/awr/n1000_eval.json'}
    names = {'base': 'base', 'ppoA6000': 'PPO on A, 6000 ep.', 'filtered_bc_r1000': 'filtered BC N=1000 (replay)',
             'sft_expert_r1000': 'SFT expert N=1000 (replay)', 'awr_r1000': 'AWR N=1000 (replay)'}
    for name, start in starts.items():
        b0 = [succ(res(start(s)), 'right') for s in seeds]
        cells = [ms([succ(res(f'{R}/m6/s{s}/{name}/n{n}_eval.json'), 'right') for s in seeds]) for n in (200, 1000, 3000)]
        a = ms([succ(res(f'{R}/m6/s{s}/{name}/n3000_eval.json'), 'left') for s in seeds])
        print(f'| {names[name]} | {ms(b0)} | ' + ' | '.join(cells) + f' | {a} |')

    # ---------------- 3. second task pair
    print('\n## 3. Second task pair: B = round nut, grasped by its ring, onto the round peg\n')
    xbase = {s: res(f'{R}/x0/base_s{s}/eval.json') for s in seeds}
    print(f"base: A {ms([succ(xbase[s], 'left') for s in seeds])}, B {ms([succ(xbase[s], 'round') for s in seeds])}\n")
    print('| arm | ΔA | ΔB (round nut) |')
    print('|---|---|---|')
    rows = [(f'PPO, {n} episodes', lambda s, n=n: f'{R}/x2/s{s}/ppo/n{n}_eval.json') for n in (1000, 3000, 6000)]
    rows += [(f'{m} N=1000 (replay)', lambda s, m=m: f'{R}/x1r/s{s}/{m}/n1000_eval.json') for m in ('filtered_bc', 'sft_expert')]
    rows += [(f'{m}, lr 1e-6, {st} steps (no replay)', lambda s, m=m, st=st: f'{R}/x5/s{s}/{m}_lr1e-06early/step{st}_eval.json')
             for m in ('sft_expert', 'filtered_bc') for st in (200, 400, 700, 1000)]
    for name, path in rows:
        ev = {s: res(path(s)) for s in seeds}
        if any(ev.values()):
            print(f'| {name} | {dms(delta(ev, xbase, "left"))} | {dms(delta(ev, xbase, "round"))} |')


if __name__ == '__main__':
    main()
