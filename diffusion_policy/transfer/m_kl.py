"""
KL from the base policy, through PPO's noisy denoising chain (transfer.kl), for every
post-trained checkpoint of the redesigned experiment: on A states with goal A (what PPO's
updates are measured on) and on B states with goal B. Observation windows come from the mixed
pretraining pool, as in transfer.m_shift.

  python -m transfer.m_kl [--seeds 0,1,2] [--n 1000] [--arms_from m_shift|kl_bc]
"""

import glob
import json
import os
import re

import click
import numpy as np
import torch

from transfer.common import goal_vec, load_checkpoint
from transfer.kl import chain_kl, cond_of, ref_chain
from transfer.m_shift import DATA, R, arms, succ_delta, windows


def kl_bc_arms(s):
    """Checkpoints of the KL-penalized BC runs (m7)."""
    out = []
    for d in sorted(glob.glob(f'{R}/m7/s{s}/*')):
        name = os.path.basename(d)
        for f in sorted(glob.glob(f'{d}/step*_eval.json'), key=lambda x: int(re.search(r'step(\d+)_', x).group(1))):
            st = int(re.search(r'step(\d+)_', f).group(1))
            ck = f'{d}/train/step{st}.ckpt'
            ck = ck if os.path.exists(ck) else f'{d}/train/final.ckpt'
            out.append((f'KL-BC {name}', f'{st} steps', ck, f))
    return out


@torch.no_grad()
def mean_kl(pol, base, obs, peg, device, bs=500):
    g = goal_vec(peg)
    o = np.concatenate([obs, np.broadcast_to(g, obs.shape[:2] + g.shape)], axis=-1).astype(np.float32)
    kls = []
    for i in range(0, len(o), bs):
        torch.manual_seed(2000 + i)                 # same chains for every checkpoint
        cond = cond_of(base, torch.from_numpy(o[i:i + bs]).to(device))
        kls.append(chain_kl(pol, cond, ref_chain(base, cond)).cpu().numpy())
    return float(np.concatenate(kls).mean())


@click.command()
@click.option('--seeds', default='0,1,2')
@click.option('--n', default=1000, help='Observation windows per goal')
@click.option('--arms_from', type=click.Choice(['m_shift', 'kl_bc']), default='m_shift')
@click.option('--out_json', default=None)
def main(seeds, n, arms_from, out_json):
    device = torch.device('cuda')
    rng = np.random.default_rng(0)
    obs_a = windows(f'{DATA}/pretrain_mixed_left.hdf5', n, rng)
    obs_b = windows(f'{DATA}/pretrain_mixed_right.hdf5', n, rng)
    rows = {}
    for s in map(int, seeds.split(',')):
        base_eval = json.load(open(f'{R}/m0/base_s{s}/eval.json'))['results']
        _, base, _ = load_checkpoint(f'{R}/m0/base_s{s}/base/final.ckpt', device)
        base.eval()
        for group, label, ckpt, ev in (arms(s) if arms_from == 'm_shift' else kl_bc_arms(s)):
            if not os.path.exists(ckpt):
                continue
            _, pol, payload = load_checkpoint(ckpt, device)
            if payload['extra_cond_dim']:
                continue
            pol.eval()
            dA, dB = succ_delta(ev, base_eval)
            row = {'seed': s, 'kl_A': mean_kl(pol, base, obs_a, 'left', device),
                   'kl_B': mean_kl(pol, base, obs_b, 'right', device), 'dA': dA, 'dB': dB}
            rows.setdefault((group, label), []).append(row)
            print(f"s{s} {group:38s} {label:12s} KL on A states {row['kl_A']:8.2f}, on B states {row['kl_B']:8.2f}",
                  flush=True)
    if out_json:
        json.dump({f'{g} | {l}': v for (g, l), v in rows.items()}, open(out_json, 'w'), indent=1)
    print('\n| arm | checkpoint | ΔA | ΔB | KL from base, A states | KL from base, B states |')
    print('|---|---|---|---|---|---|')
    for (g, l), v in rows.items():
        m = lambda k: np.mean([x[k] for x in v if x[k] is not None]) if any(x[k] is not None for x in v) else np.nan
        print(f"| {g} | {l} | {m('dA'):+.2f} | {m('dB'):+.2f} | {m('kl_A'):.2f} | {m('kl_B'):.2f} |")


if __name__ == '__main__':
    main()
