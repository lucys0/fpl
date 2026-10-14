"""
Episodes, task splits and training windows for the K-task experiments.

Task splits of the 8 x 4 grid (object o, target t = o + j mod 4), a Latin design so every object and every
target is in every split:
  U (j = 0): never in pretraining - zero-shot and few-shot learnability;
  H (j = 1): in pretraining, never in RL - the fixed held-out set every RL run is measured on;
  R (j = 2, 3): in pretraining and eligible for RL - the K training tasks are drawn from these 16.

Episode files (.npz): obs (sum T, 40) and action (sum T, 10) concatenated over episodes; per episode
ep_start, ep_len, task, q, success, first_success_step, seed, and optional extra per-episode arrays.
"""

import json

import numpy as np
import torch

from ktask.env import N_OBJ, N_TASKS, N_TGT, OBJ_NAMES, TGT_NAMES

HORIZON, N_OBS, N_ACT = 16, 2, 8


def _split(j):
    return sorted(o * N_TGT + (o + j) % N_TGT for o in range(N_OBJ))


SPLITS = {'U': _split(0), 'H': _split(1), 'R': sorted(_split(2) + _split(3))}


def parse_tasks(s):
    """'all', split letters ('R', 'RH', 'H,U'), task ids or names (cube-goal), comma-separated."""
    if isinstance(s, (list, tuple)):
        return [int(t) for t in s]
    if s == 'all':
        return list(range(N_TASKS))
    out = []
    for part in s.split(','):
        part = part.strip()
        if part and all(c in SPLITS for c in part):
            for c in part:
                out += SPLITS[c]
        elif '-' in part:
            o, t = part.split('-')
            out.append(OBJ_NAMES.index(o) * N_TGT + TGT_NAMES.index(t))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def save_episodes(path, eps, meta=None):
    lens = np.array([len(e['action']) for e in eps], dtype=np.int64)
    arrays = dict(obs=np.concatenate([e['obs'] for e in eps]).astype(np.float32),
                  action=np.concatenate([e['action'] for e in eps]).astype(np.float32),
                  ep_len=lens, ep_start=np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64))
    skip = {'obs', 'action'}
    for k in eps[0]:
        if k not in skip:
            arrays[k] = np.array([e[k] for e in eps])
    arrays['meta'] = np.array(json.dumps(meta or {}))
    tmp = path + '.tmp.npz'
    np.savez_compressed(tmp, **arrays)
    import os
    os.replace(tmp, path)


def load_episodes(path):
    """Dict of arrays (see the module docstring)."""
    with np.load(path) as f:
        d = {k: f[k] for k in f.files}
    d['meta'] = json.loads(str(d['meta']))
    return d


def select(d, tasks=None, success_only=False, first_n=None, fastest_frac=None):
    """Indices of episodes: in `tasks`, the first `first_n` of each task (before the success filter, so
    smaller budgets are prefixes of larger ones), successful only, and optionally only the `fastest_frac`
    share of successful episodes with the shortest lengths (pooled over tasks)."""
    idx = np.arange(len(d['ep_len']))
    if tasks is not None:
        idx = idx[np.isin(d['task'][idx], tasks)]
    if first_n is not None:
        keep = []
        for t in np.unique(d['task'][idx]):
            keep += list(idx[d['task'][idx] == t][:first_n])
        idx = np.array(sorted(keep), dtype=np.int64)
    if success_only:
        idx = idx[d['success'][idx]]
    if fastest_frac is not None:
        assert success_only, 'fastest_frac ranks successful episodes'
        k = max(1, round(fastest_frac * len(idx)))
        idx = np.sort(idx[np.argsort(d['ep_len'][idx], kind='stable')[:k]])
    return idx


class Windows:
    """Training windows on the GPU: obs (B, N_OBS, 40), action (B, HORIZON, 10), weight (B,); windows start
    N_OBS - 1 steps before an episode and end up to N_ACT - 1 steps after it, edges repeated (as
    diffusion_policy's SequenceSampler with pad_before = N_OBS - 1, pad_after = N_ACT - 1)."""

    def __init__(self, parts, device):
        """parts: list of (data dict, episode indices, per-step weights or None, repeat count)."""
        obs, act, wts, rows = [], [], [], []
        offset = 0
        for d, idx, w, rep in parts:
            for j, e in enumerate(idx):
                s, T = int(d['ep_start'][e]), int(d['ep_len'][e])
                obs.append(d['obs'][s:s + T])
                act.append(d['action'][s:s + T])
                wts.append(np.ones(T, np.float32) if w is None else np.asarray(w[j], np.float32))
                starts = np.arange(-(N_OBS - 1), max(T - HORIZON + N_ACT - 1, -(N_OBS - 1)) + 1)
                r = np.stack([starts, np.full_like(starts, offset), np.full_like(starts, T)], 1)
                rows += [r] * rep
                offset += T
        self.obs = torch.from_numpy(np.concatenate(obs)).to(device)
        self.act = torch.from_numpy(np.concatenate(act)).to(device)
        self.w = torch.from_numpy(np.concatenate(wts)).to(device)
        self.rows = torch.from_numpy(np.concatenate(rows)).to(device)
        self.device = device

    def __len__(self):
        return len(self.rows)

    def sample(self, batch_size, gen):
        r = self.rows[torch.randint(len(self.rows), (batch_size,), device=self.device, generator=gen)]
        t = r[:, :1] + torch.arange(HORIZON, device=self.device)[None]
        t = t.clamp(min=0).minimum(r[:, 2:3] - 1) + r[:, 1:2]
        cur = (r[:, 0] + N_OBS - 1).clamp(min=0).minimum(r[:, 2] - 1) + r[:, 1]
        return self.obs[t[:, :N_OBS]], self.act[t], self.w[cur]
