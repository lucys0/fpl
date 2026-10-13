"""
Train or fine-tune a goal-conditioned flow-transformer policy for the
two-peg transfer experiments. One JSON spec per run:

  {"out_dir": ..., "init_ckpt": null or path,
   "method": "bc" | "awr" | "udrl",          # SFT and filtered BC are both "bc"
   "udrl_mode": "scalar" | "axes",           # udrl only
   "data": [{"path": ..., "pegs": ["left"], "max_per_peg": null, "first_n": null,
             "success_only": false, "rng_seed": 0, "replay": false}, ...],
   "replay_frac": 0.5,                       # share of training steps from "replay" data (repeated to reach it)
   "steps": 20000, "lr": 2e-4, "warmup": 500, "batch_size": 1024, "seed": 0,
   "awr": {"gamma": 0.99, "beta": 1.0, "max_weight": 20.0},
   "wandb": {"project": ..., "group": ..., "name": ...}}

Usage: python -m transfer.train spec.json
"""

import copy
import json
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from diffusion_policy.common.normalize_util import array_to_stats, get_identity_normalizer_from_stat
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.diffusion.ema_model import EMAModel

from transfer.common import (GOAL_DIM, MAX_STEPS, EpisodeDataset, build_policy, goal_vec,
                             load_checkpoint, load_episodes, save_checkpoint, widen_cond)


def select_episodes(data_specs):
    """first_n is applied before success_only, so smaller budgets are prefixes of larger ones.
    Episodes from specs marked "replay" carry attrs['replay'] = True."""
    episodes = []
    for spec in data_specs:
        eps = load_episodes(spec['path'], pegs=spec.get('pegs'), max_per_peg=spec.get('max_per_peg'),
                            rng_seed=spec.get('rng_seed', 0))
        if spec.get('first_n') is not None:
            eps = eps[:spec['first_n']]
        if spec.get('success_only'):
            eps = [e for e in eps if e['attrs']['success']]
        for e in eps:
            e['attrs']['replay'] = bool(spec.get('replay', False))
        episodes += eps
    return episodes


def mix_replay(episodes, weights, frac, log):
    """Repeat the replay episodes so they make up about `frac` of all training steps."""
    new = [i for i, e in enumerate(episodes) if not e['attrs']['replay']]
    old = [i for i, e in enumerate(episodes) if e['attrs']['replay']]
    if not old or not new:
        return episodes, weights
    steps = lambda idx: sum(len(episodes[i]['action']) for i in idx)
    reps = max(1, round(frac / (1 - frac) * steps(new) / steps(old)))
    order = new + old * reps
    log(f"replay: {len(old)} episodes x{reps} -> {reps * steps(old) / (reps * steps(old) + steps(new)):.0%} of steps")
    return [episodes[i] for i in order], (None if weights is None else [weights[i] for i in order])


# ---------------------------------------------------------------- AWR

def awr_weights(episodes, gamma, beta, max_weight, device, log):
    """Per-step weights exp(A / (beta * std A)), A = Monte-Carlo return - V(s).

    Reward is 1 on the step the nut lands on the commanded peg (recorded
    successful rollouts end there) and 0 otherwise, discounted by gamma.
    """
    obs, ret = [], []
    for ep in episodes:
        T = len(ep['action'])
        g = np.zeros(T, dtype=np.float32)
        if ep['attrs']['success']:
            g = gamma ** np.arange(T - 1, -1, -1, dtype=np.float32)
        obs.append(np.concatenate([ep['obs'][:T], np.broadcast_to(goal_vec(ep['attrs']['target_peg']), (T, GOAL_DIM))], -1))
        ret.append(g)
    X = torch.from_numpy(np.concatenate(obs)).to(device)
    Y = torch.from_numpy(np.concatenate(ret)).to(device)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    value = nn.Sequential(nn.Linear(X.shape[1], 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                          nn.Linear(256, 1)).to(device)
    opt = torch.optim.Adam(value.parameters(), lr=1e-3)
    for step in range(3000):
        idx = torch.randint(0, len(X), (4096,), device=device)
        loss = ((value((X[idx] - mu) / sd).squeeze(-1) - Y[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        adv = (Y - value((X - mu) / sd).squeeze(-1)).cpu().numpy()
    scale = beta * (adv.std() + 1e-6)
    w = np.minimum(np.exp(adv / scale), max_weight)
    log(f"AWR: value mse {loss.item():.4f}, advantage std {adv.std():.3f}, "
        f"weights mean {w.mean():.2f} max {w.max():.1f} (clipped {np.mean(w >= max_weight):.1%})")
    out, i = [], 0
    for ep in episodes:
        T = len(ep['action'])
        out.append(w[i:i + T]); i += T
    return out


# ---------------------------------------------------------------- upside-down RL

def udrl_condition(ep, mode):
    s = float(ep['attrs']['success'])
    speed = s * (1.0 - len(ep['action']) / MAX_STEPS)     # successful rollouts end at success
    return [0.5 * (s + speed)] if mode == 'scalar' else [s, speed]


def udrl_eval_target(episodes, mode):
    """Condition used at test time: a high but in-distribution value."""
    succ = [ep for ep in episodes if ep['attrs']['success']]
    vals = np.array([udrl_condition(ep, mode) for ep in succ]) if succ else np.ones((1, 1 if mode == 'scalar' else 2))
    return np.percentile(vals, 90, axis=0).tolist()


# ---------------------------------------------------------------- main

def main(spec_path):
    spec = json.load(open(spec_path))
    out_dir = spec['out_dir']
    os.makedirs(out_dir, exist_ok=True)
    log_f = open(os.path.join(out_dir, 'train_log.txt'), 'a')

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True); log_f.write(line + '\n'); log_f.flush()

    seed = spec.get('seed', 0)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    device = torch.device('cuda')
    method = spec['method']

    episodes = select_episodes(spec['data'])
    n_succ = sum(bool(e['attrs']['success']) for e in episodes)
    log(f"{method}: {len(episodes)} episodes ({n_succ} successful), "
        f"{sum(len(e['action']) for e in episodes)} steps, pegs "
        f"{dict(zip(*np.unique([e['attrs']['target_peg'] for e in episodes], return_counts=True)))}")

    if not episodes:
        # e.g. filtered BC when no rollout succeeded: nothing to learn from, so the policy stays as it was
        assert spec.get('init_ckpt'), 'training from scratch needs data'
        shutil.copy(spec['init_ckpt'], os.path.join(out_dir, 'final.ckpt'))
        log(f"no training data; final.ckpt = unchanged copy of {spec['init_ckpt']}")
        return

    extra_dim, cond_fn, info = 0, None, {'spec': spec}
    if method == 'udrl':
        mode = spec['udrl_mode']
        extra_dim = 1 if mode == 'scalar' else 2
        cond_fn = lambda ep: udrl_condition(ep, mode)
        info['eval_cond'] = udrl_eval_target(episodes, mode)
        log(f"UDRL ({mode}) eval condition: {info['eval_cond']}")

    if spec.get('init_ckpt'):
        policy, ema_policy, payload = load_checkpoint(spec['init_ckpt'])
        if payload['extra_cond_dim'] != extra_dim:      # e.g. UDRL fine-tuning of a goal-only policy
            assert payload['extra_cond_dim'] == 0, "can only widen a goal-only policy"
            policy, ema_policy = widen_cond(policy, extra_dim), widen_cond(ema_policy, extra_dim)
    else:
        policy = build_policy(extra_dim)
        ema_policy = None

    weights = None
    if method == 'awr':
        a = spec.get('awr', {})
        rollouts = [e for e in episodes if not e['attrs']['replay']]
        w = iter(awr_weights(rollouts, a.get('gamma', 0.99), a.get('beta', 1.0), a.get('max_weight', 20.0), device, log))
        weights = [np.ones(len(e['action']), dtype=np.float32) if e['attrs']['replay'] else next(w) for e in episodes]
    episodes, weights = mix_replay(episodes, weights, spec.get('replay_frac', 0.5), log)

    dataset = EpisodeDataset(episodes, cond_fn=cond_fn, weights=weights)
    if not spec.get('init_ckpt'):
        policy.set_normalizer(dataset.get_normalizer())
        ema_policy = copy.deepcopy(policy)
    if weights is not None:
        # compute_loss normalizes every batch key, so weights need a pass-through entry
        stat = array_to_stats(np.concatenate(weights)[:, None])
        for p in (policy, ema_policy):
            p.normalizer['weight'] = get_identity_normalizer_from_stat(stat)
    policy.to(device); ema_policy.to(device)

    # a dataset smaller than one batch (e.g. a single demo) trains full-batch instead of yielding no batches
    batch_size = spec.get('batch_size', 1024)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                                         persistent_workers=True, drop_last=len(dataset) >= batch_size)
    steps = spec['steps']
    optimizer = policy.get_optimizer(weight_decay=1e-3, learning_rate=spec.get('lr', 2e-4), betas=(0.9, 0.95))
    scheduler = get_scheduler('cosine', optimizer=optimizer, num_warmup_steps=spec.get('warmup', 500),
                              num_training_steps=steps)
    ema = EMAModel(ema_policy, update_after_step=0, inv_gamma=1.0, power=0.75, min_value=0.0, max_value=0.9999)

    wb = None
    if spec.get('wandb'):
        try:
            import wandb
            wb = wandb.init(config=spec, dir=out_dir, **spec['wandb'])
        except Exception as e:                       # logging must never kill a run
            log(f"wandb disabled: {e!r}")

    # jobs on the iris partition can be preempted and requeued: resume from the last snapshot
    step, t0, losses = 0, time.time(), []
    resume_path = os.path.join(out_dir, 'resume.ckpt')
    resume_every = spec.get('resume_every', 1000)
    if os.path.exists(resume_path):
        r = torch.load(resume_path, map_location=device)
        policy.load_state_dict(r['policy']); ema_policy.load_state_dict(r['ema'])
        optimizer.load_state_dict(r['optimizer']); scheduler.load_state_dict(r['scheduler'])
        step, ema.optimization_step = r['step'], r['ema_step']
        log(f"resumed from step {step}")
    save_every = spec.get('save_every')
    policy.train()
    while step < steps:
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            loss = policy.compute_loss(batch)
            optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()
            ema.step(policy)
            losses.append(loss.item()); step += 1
            if step % 500 == 0 or step == steps:
                log(f"step {step}/{steps} loss {np.mean(losses):.4f} lr {scheduler.get_last_lr()[0]:.2e} "
                    f"({step / (time.time() - t0):.1f} it/s)")
                if wb: wb.log({'train_loss': np.mean(losses), 'lr': scheduler.get_last_lr()[0]}, step=step)
                losses = []
            if step % resume_every == 0 and step < steps:
                torch.save({'policy': policy.state_dict(), 'ema': ema_policy.state_dict(),
                            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                            'step': step, 'ema_step': ema.optimization_step}, resume_path + '.tmp')
                os.replace(resume_path + '.tmp', resume_path)
            if save_every and step % save_every == 0 and step < steps:
                save_checkpoint(os.path.join(out_dir, f'step{step}.ckpt'), policy, ema_policy, extra_dim, info)
            if step >= steps:
                break
    save_checkpoint(os.path.join(out_dir, 'final.ckpt'), policy, ema_policy, extra_dim, info)
    if os.path.exists(resume_path):
        os.remove(resume_path)
    log(f"saved {os.path.join(out_dir, 'final.ckpt')} after {(time.time() - t0) / 60:.1f} min")
    if wb: wb.finish()


if __name__ == '__main__':
    main(sys.argv[1])
