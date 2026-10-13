"""
transfer.m_shift for the second task pair (B = the round nut, grasped by its ring): how far each
checkpoint moves the commanded positions on A states (square nut, goal A) and on B states (round nut,
goal B), and whether, on the same A states, the change under goal A and under goal B point the same way.

  python -m transfer.x_shift [--seeds 0,1,2] [--n 2000]
"""

import json
import os

import click
import numpy as np
import torch

from transfer.common import load_checkpoint
from transfer.m_shift import DATA, R, actions, windows


def arms(s):
    out = [('PPO', f'{n} ep.', f'{R}/x2/s{s}/ppo/ppo/ppo_ep{n}.ckpt', f'{R}/x2/s{s}/ppo/n{n}_eval.json')
           for n in (1000, 3000, 6000)]
    for m in ('sft_expert', 'filtered_bc'):
        d = f'{R}/x5/s{s}/{m}_lr1e-06early'
        out += [(f'{m} lr 1e-6 (no replay)', f'{st} steps', f'{d}/train/' + ('final.ckpt' if st == 1000 else f'step{st}.ckpt'),
                 f'{d}/step{st}_eval.json') for st in (200, 400, 1000)]
        out.append((f'{m} N=1000 (replay)', 'final', f'{R}/x1r/s{s}/{m}/n1000/final.ckpt', f'{R}/x1r/s{s}/{m}/n1000_eval.json'))
    return out


@click.command()
@click.option('--seeds', default='0,1,2')
@click.option('--n', default=2000)
def main(seeds, n):
    device = torch.device('cuda')
    rng = np.random.default_rng(0)
    obs_a = windows(f'{DATA}/pretrain_mixed_left.hdf5', n, rng)
    obs_b = windows(f'{DATA}/pretrain_mixed_round.hdf5', n, rng)
    rows = {}
    for s in map(int, seeds.split(',')):
        be = json.load(open(f'{R}/x0/base_s{s}/eval.json'))['results']
        _, base, _ = load_checkpoint(f'{R}/x0/base_s{s}/base/final.ckpt', device)
        base.eval()
        b_aa, b_bb, b_ab = actions(base, obs_a, 'left', device), actions(base, obs_b, 'round', device), \
            actions(base, obs_a, 'round', device)
        for g, l, ckpt, ev in arms(s):
            if not os.path.exists(ckpt):
                continue
            _, pol, _ = load_checkpoint(ckpt, device)
            pol.eval()
            aa, bb, ab = actions(pol, obs_a, 'left', device), actions(pol, obs_b, 'round', device), \
                actions(pol, obs_a, 'round', device)
            dist = lambda x, y: float(np.linalg.norm(x - y, axis=-1).mean() * 100)
            ga, gb = (aa - b_aa).reshape(n, -1), (ab - b_ab).reshape(n, -1)
            cos = float(((ga * gb).sum(1) / (np.linalg.norm(ga, axis=1) * np.linalg.norm(gb, axis=1) + 1e-9)).mean())
            r = json.load(open(ev))['results'] if os.path.exists(ev) else None
            rows.setdefault((g, l), []).append({
                'shift_A': dist(aa, b_aa), 'shift_B': dist(bb, b_bb), 'cos': cos,
                'dA': r['left']['success'] - be['left']['success'] if r else None,
                'dB': r['round']['success'] - be['round']['success'] if r else None})
    print('| arm | checkpoint | seeds | ΔA | ΔB (round) | change on A states (cm) | on B states (cm) | '
          'cos(change with goal A, with goal B) |')
    print('|---|---|---|---|---|---|---|---|')
    for (g, l), v in rows.items():
        m = lambda k: np.mean([x[k] for x in v if x[k] is not None]) if any(x[k] is not None for x in v) else np.nan
        print(f"| {g} | {l} | {len(v)} | {m('dA'):+.2f} | {m('dB'):+.2f} | {m('shift_A'):.2f} | {m('shift_B'):.2f} | "
              f"{m('cos'):+.2f} |")


if __name__ == '__main__':
    main()
