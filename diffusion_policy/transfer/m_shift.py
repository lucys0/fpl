"""
How post-training on A changes the policy's actions, on A states vs B states.

Fixed observation windows come from the mixed pretraining pool: left-peg episodes queried
with goal A, right-peg episodes with goal B. Every checkpoint's action chunk is computed with
the same initial noise as its seed's base policy (EMA weights, as in evaluation), and the
change is the mean distance of the commanded end-effector positions from the base's, in cm,
over the executed 8 steps. On the A states, the change is also measured with the goal
switched to B, and the cosine between the changes under the two goals tells whether the
update is shared across goals (1) or specific to the trained one (0 or below).

  python -m transfer.m_shift [--seeds 0,1,2] [--n 2000]
"""

import json
import os

import click
import numpy as np
import torch

from transfer.common import N_OBS_STEPS, goal_vec, load_checkpoint, load_episodes

R = '/iris/u/lucyshi/fpl_transfer/runs'
DATA = '/iris/u/lucyshi/fpl_transfer/data'


def windows(path, n, rng):
    eps = load_episodes(path)
    out = []
    for _ in range(n):
        e = eps[rng.integers(len(eps))]
        t = rng.integers(N_OBS_STEPS - 1, len(e['obs']))
        out.append(e['obs'][t - N_OBS_STEPS + 1:t + 1])
    return np.stack(out)


def actions(policy, obs, peg, device, bs=1000):
    g = goal_vec(peg)
    o = np.concatenate([obs, np.broadcast_to(g, obs.shape[:2] + g.shape)], axis=-1).astype(np.float32)
    out = []
    for i in range(0, len(o), bs):
        torch.manual_seed(1000 + i)                 # same initial noise for every checkpoint
        with torch.no_grad():
            a = policy.predict_action({'obs': torch.from_numpy(o[i:i + bs]).to(device)})['action']
        out.append(a[..., :3].cpu().numpy())        # commanded end-effector positions
    return np.concatenate(out)                      # (N, 8, 3)


def succ_delta(eval_path, base_eval):
    if not os.path.exists(eval_path):
        return None, None
    r = json.load(open(eval_path))['results']
    return r['left']['success'] - base_eval['left']['success'], r['right']['success'] - base_eval['right']['success']


def arms(s):
    """(group, label, checkpoint, eval json)"""
    out = [('PPO', f'{n} ep.', f'{R}/m2/s{s}/ppo/ppo/ppo_ep{n}.ckpt', f'{R}/m2/s{s}/ppo/n{n}_eval.json')
           for n in (200, 1000, 3000, 6000)]
    for m in ('sft_expert', 'filtered_bc'):
        d = f'{R}/m5/s{s}/{m}_lr1e-06early'
        out += [(f'{m} offline lr 1e-6', f'{st} steps', f'{d}/train/' + ('final.ckpt' if st == 1000 else f'step{st}.ckpt'),
                 f'{d}/step{st}_eval.json') for st in (100, 200, 400, 700, 1000)]
        d = f'{R}/m5/s{s}/{m}_lr1e-06'
        out += [(f'{m} offline lr 1e-6', f'{st} steps', f'{d}/train/' + ('final.ckpt' if st == 10000 else f'step{st}.ckpt'),
                 f'{d}/step{st}_eval.json') for st in (3000, 10000)]
        d = f'{R}/m5/s{s}/{m}_lr1e-04'
        out += [(f'{m} offline lr 1e-4', f'{st} steps', f'{d}/train/' + ('final.ckpt' if st == 1000 else f'step{st}.ckpt'),
                 f'{d}/step{st}_eval.json') for st in (100, 1000)]
    d = f'{R}/m4/s{s}/filtered_bc_lr1e-06'
    out += [('filtered_bc on own rollouts lr 1e-6', f'{200 * (r + 1)} ep.', f'{d}/r{r}/final.ckpt',
             f'{d}/n{200 * (r + 1)}_eval.json') for r in (0, 2, 4)]
    for stage, tag in (('m1r', 'replay'), ('m1', 'no replay')):
        out += [(f'offline N=1000, {tag}', m, f'{R}/{stage}/s{s}/{m}/n1000/final.ckpt', f'{R}/{stage}/s{s}/{m}/n1000_eval.json')
                for m in ('sft_pool', 'filtered_bc', 'awr', 'sft_expert')]
    return out


@click.command()
@click.option('--seeds', default='0,1,2')
@click.option('--n', default=2000, help='Observation windows per goal')
@click.option('--out_json', default='/iris/u/lucyshi/fpl_transfer/results/m_shift.json')
def main(seeds, n, out_json):
    device = torch.device('cuda')
    rng = np.random.default_rng(0)
    obs_a = windows(f'{DATA}/pretrain_mixed_left.hdf5', n, rng)
    obs_b = windows(f'{DATA}/pretrain_mixed_right.hdf5', n, rng)
    rows = {}
    for s in map(int, seeds.split(',')):
        base_eval = json.load(open(f'{R}/m0/base_s{s}/eval.json'))['results']
        _, base, _ = load_checkpoint(f'{R}/m0/base_s{s}/base/final.ckpt', device)
        base.eval()
        b_aa, b_bb, b_ab = actions(base, obs_a, 'left', device), actions(base, obs_b, 'right', device), \
            actions(base, obs_a, 'right', device)
        for group, label, ckpt, ev in arms(s):
            if not os.path.exists(ckpt):
                continue
            _, pol, payload = load_checkpoint(ckpt, device)
            if payload['extra_cond_dim']:
                continue
            pol.eval()
            aa, bb, ab = actions(pol, obs_a, 'left', device), actions(pol, obs_b, 'right', device), \
                actions(pol, obs_a, 'right', device)
            dist = lambda x, y: float(np.linalg.norm(x - y, axis=-1).mean() * 100)
            ga, gb = (aa - b_aa).reshape(n, -1), (ab - b_ab).reshape(n, -1)
            cos = (ga * gb).sum(1) / (np.linalg.norm(ga, axis=1) * np.linalg.norm(gb, axis=1) + 1e-9)
            dA, dB = succ_delta(ev, base_eval)
            rows.setdefault((group, label), []).append(
                {'seed': s, 'shift_A': dist(aa, b_aa), 'shift_B': dist(bb, b_bb), 'shift_A_states_goal_B': dist(ab, b_ab),
                 'cos_goalA_goalB': float(cos.mean()), 'dA': dA, 'dB': dB})
            print(f"s{s} {group:38s} {label:12s} shift A {rows[(group, label)][-1]['shift_A']:.2f} cm, "
                  f"B {rows[(group, label)][-1]['shift_B']:.2f} cm, cos {cos.mean():+.2f}", flush=True)
    json.dump({f'{g} | {l}': v for (g, l), v in rows.items()}, open(out_json, 'w'), indent=1)

    print('\n| arm | checkpoint | ΔA | ΔB | action change on A states (cm) | on B states (cm) | '
          'B/A change | cos(change with goal A, with goal B) on A states |')
    print('|---|---|---|---|---|---|---|---|')
    for (g, l), v in rows.items():
        m = lambda k: np.mean([x[k] for x in v if x[k] is not None]) if any(x[k] is not None for x in v) else np.nan
        print(f"| {g} | {l} | {m('dA'):+.2f} | {m('dB'):+.2f} | {m('shift_A'):.2f} | {m('shift_B'):.2f} | "
              f"{m('shift_B') / m('shift_A'):.2f} | {m('cos_goalA_goalB'):+.2f} |")


if __name__ == '__main__':
    main()
