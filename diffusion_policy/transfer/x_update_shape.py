"""
What kind of change each post-training method makes (second pair, seeds 0-2):
  constancy - on A and B pool states, how close the change of planned positions is to one constant offset:
              |mean change| / mean |change| (1 = the same shift everywhere, ~0 = state-dependent)
  layers    - relative parameter change ||theta - theta_base|| / ||theta_base|| by module group

  python -m transfer.x_update_shape
"""
import numpy as np
import torch

from transfer.common import load_checkpoint
from transfer.m_shift import DATA, R, actions, windows

ARMS = {'PPO 6000 ep.': '{R}/x2/s{s}/ppo/ppo/ppo_ep6000.ckpt',
        'SFT expert, replay': '{R}/x1r/s{s}/sft_expert/n1000/final.ckpt',
        'SFT expert, lr 1e-6, 400 steps': '{R}/x5/s{s}/sft_expert_lr1e-06early/train/step400.ckpt'}


def group(name):
    for g in ('cond_obs_emb', 'time_emb', 'input_emb', 'encoder', 'decoder', 'ln_f', 'head'):
        if g in name:
            return g
    return 'other'


def main():
    device = torch.device('cuda')
    rng = np.random.default_rng(0)
    obs = {'A': windows(f'{DATA}/pretrain_mixed_left.hdf5', 2000, rng), 'B': windows(f'{DATA}/pretrain_mixed_round.hdf5', 2000, rng)}
    peg = {'A': 'left', 'B': 'round'}
    const, layers = {}, {}
    for s in (0, 1, 2):
        _, base, _ = load_checkpoint(f'{R}/x0/base_s{s}/base/final.ckpt', device)
        base.eval()
        bp = {k: actions(base, obs[k], peg[k], device) for k in obs}
        bstate = {n: p.detach() for n, p in base.model.named_parameters()}
        for arm, path in ARMS.items():
            _, pol, _ = load_checkpoint(path.format(R=R, s=s), device)
            pol.eval()
            for k in obs:
                d = (actions(pol, obs[k], peg[k], device) - bp[k]).mean(1)[:, :2]      # (N, 2) world xy, cm below
                const.setdefault((arm, k), []).append((np.linalg.norm(d.mean(0)) / np.linalg.norm(d, axis=1).mean(),
                                                       d.mean(0) * 100))
            for n, p in pol.model.named_parameters():
                g = group(n)
                num, den = layers.setdefault((arm, g), [0.0, 0.0])
                layers[(arm, g)] = [num + (p.detach() - bstate[n]).pow(2).sum().item(), den + bstate[n].pow(2).sum().item()]
    print('| arm | states | constancy (1 = one shift everywhere) | mean shift of planned xy (cm), per seed |')
    print('|---|---|---|---|')
    for (arm, k), v in const.items():
        print(f"| {arm} | {k} | {np.mean([c for c, _ in v]):.2f} | " + ', '.join(f'({m[0]:+.2f}, {m[1]:+.2f})' for _, m in v) + ' |')
    groups = sorted({g for _, g in layers})
    print('\n| arm | ' + ' | '.join(groups) + ' |')
    print('|---|' + '---|' * len(groups))
    for arm in ARMS:
        print(f'| {arm} | ' + ' | '.join(f'{np.sqrt(layers[(arm, g)][0] / layers[(arm, g)][1]):.1e}' for g in groups) + ' |')


if __name__ == '__main__':
    main()
