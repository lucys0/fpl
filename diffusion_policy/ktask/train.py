"""
Train or fine-tune the flow policy on recorded episodes. One JSON spec per run:

  {"out_dir": ..., "init_ckpt": null or path,
   "method": "bc" | "awr",                     # SFT and filtered BC are both "bc"
   "data": [{"path": ..., "tasks": "RH" or [ids], "first_n": null, "success_only": false,
             "fastest_frac": null, "replay": false}, ...],
   "replay_frac": 0.5,                         # share of each batch drawn from "replay" data
   "steps": 40000, "lr": 2e-4, "warmup": 500, "batch_size": 1024, "seed": 0, "save_at": [...],
   "awr": {"gamma": 0.99, "beta": 1.0, "max_weight": 20.0, "time_cost": 0.0},   # gamma per env step
   "wandb": {"project": ..., "group": ..., "name": ...}}

A fresh policy fits its normalizers to the training data; a fine-tuned one keeps its checkpoint's, and
starts from the checkpoint's (EMA) weights. Saves the EMA weights as step<N>.ckpt at save_at and final.ckpt.
Resumes from resume.pt after preemption.

  python -m ktask.train spec.json
"""

import json
import os
import sys
import time

import numpy as np
import torch

from ktask.data import Windows, load_episodes, parse_tasks, select
from ktask.policy import EMA, FlowPolicy, cosine_lr, load_checkpoint, optimizer_for, save_checkpoint


def awr_weights(d, idx, gamma, beta, max_weight, time_cost, device, log, max_steps=200):
    """Per-step weights exp(A / beta) (clipped), A = discounted return-to-go minus a value baseline fitted
    on the same steps (an MLP on the observation). Reward: +1 on the last step of a successful episode,
    -time_cost / max_steps every step."""
    obs, ret = [], []
    for e in idx:
        s, T = int(d['ep_start'][e]), int(d['ep_len'][e])
        r = np.full(T, -time_cost / max_steps, np.float32)
        if d['success'][e]:
            r[-1] += 1.0
        g, out = 0.0, np.zeros(T, np.float32)
        for t in range(T - 1, -1, -1):
            g = r[t] + gamma * g
            out[t] = g
        obs.append(d['obs'][s:s + T])
        ret.append(out)
    X = torch.from_numpy(np.concatenate(obs)).to(device)
    Y = torch.from_numpy(np.concatenate(ret)).to(device)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    v = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 256), torch.nn.ReLU(), torch.nn.Linear(256, 256),
                            torch.nn.ReLU(), torch.nn.Linear(256, 1)).to(device)
    opt = torch.optim.Adam(v.parameters(), lr=1e-3)
    for it in range(3000):
        b = torch.randint(len(X), (4096,), device=device)
        loss = (v((X[b] - mu) / sd).squeeze(-1) - Y[b]).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        A = Y - v((X - mu) / sd).squeeze(-1)
        w = torch.exp(A / beta).clamp(max=max_weight).cpu().numpy()
    log(f'AWR: value loss {loss.item():.4f}, advantage std {A.std().item():.3f}, weight mean {w.mean():.2f}, '
        f'clipped {np.mean(w >= max_weight):.1%}')
    out, k = [], 0
    for r in ret:
        out.append(w[k:k + len(r)])
        k += len(r)
    return out


def main(spec_path):
    spec = json.load(open(spec_path))
    out_dir = spec['out_dir']
    os.makedirs(out_dir, exist_ok=True)
    log_f = open(os.path.join(out_dir, 'train_log.txt'), 'a')

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_f.write(line + '\n')
        log_f.flush()

    seed = spec.get('seed', 0)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True       # supervised training only; PPO's log-ratios stay in fp32
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    groups = {False: [], True: []}
    n_eps = {False: 0, True: 0}
    for ds in spec['data']:
        d = load_episodes(ds['path'])
        tasks = parse_tasks(ds['tasks']) if ds.get('tasks') is not None else None
        idx = select(d, tasks, ds.get('success_only', False), ds.get('first_n'), ds.get('fastest_frac'))
        w = None
        if spec.get('method') == 'awr' and not ds.get('replay'):
            a = spec.get('awr', {})
            w = awr_weights(d, idx, a.get('gamma', 0.99), a.get('beta', 1.0), a.get('max_weight', 20.0),
                            a.get('time_cost', 0.0), device, log)
        groups[bool(ds.get('replay'))].append((d, idx, w, 1))
        n_eps[bool(ds.get('replay'))] += len(idx)
        log(f"data {ds['path']}: {len(idx)} episodes ({d['success'][idx].mean():.2f} successful)"
            + (' [replay]' if ds.get('replay') else ''))
    win = {k: Windows(v, device) for k, v in groups.items() if v}
    replay_frac = spec.get('replay_frac', 0.5) if True in win and False in win else (1.0 if False not in win else 0.0)
    log(f"windows: {', '.join(f'{len(w)} {k}' for k, w in (('new', win.get(False)), ('replay', win.get(True))) if w)}")

    if spec.get('init_ckpt'):
        policy, _ = load_checkpoint(spec['init_ckpt'], device)
    else:
        policy = FlowPolicy().to(device)
        all_obs = torch.cat([w.obs for w in win.values()])
        all_act = torch.cat([w.act for w in win.values()])
        policy.fit_normalizers(all_obs, all_act)
    ema = EMA(policy)
    steps, bs, lr = spec['steps'], spec.get('batch_size', 1024), spec.get('lr', 2e-4)
    opt = optimizer_for(policy, lr)
    step = 0
    resume = os.path.join(out_dir, 'resume.pt')
    if os.path.exists(resume):
        r = torch.load(resume, map_location=device, weights_only=False)
        policy.load_state_dict(r['policy'])
        ema.model.load_state_dict(r['ema'])
        opt.load_state_dict(r['opt'])
        step, ema.step_count = r['step'], r['ema_step']
        gen.set_state(r['gen'].cpu() if hasattr(r['gen'], 'cpu') else r['gen'])
        log(f'resumed at step {step}')

    wb = None
    if spec.get('wandb'):
        try:
            import wandb
            wb = wandb.init(config=spec, dir=out_dir, resume='allow', **spec['wandb'])
        except Exception as e:
            log(f'wandb disabled: {e!r}')

    save_at = sorted(set(spec.get('save_at', [])) | {steps})
    n_rep = int(round(bs * replay_frac))
    policy.train()
    losses, t0 = [], time.time()
    while step < steps:
        for g in opt.param_groups:
            g['lr'] = lr * cosine_lr(step, steps, spec.get('warmup', 500))
        parts = []
        if n_rep < bs:
            parts.append(win[False].sample(bs - n_rep, gen))
        if n_rep > 0:
            parts.append(win[True].sample(n_rep, gen))
        obs, act, w = (torch.cat(x) for x in zip(*parts))
        loss = policy.loss(obs, act, w if spec.get('method') == 'awr' else None)
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema.step(policy)
        losses.append(loss.item())
        step += 1
        if step % 500 == 0 or step == steps:
            log(f'step {step}/{steps} loss {np.mean(losses):.4f} lr {opt.param_groups[0]["lr"]:.2e} '
                f'({step / (time.time() - t0):.1f} it/s)')
            if wb:
                wb.log({'loss': np.mean(losses), 'lr': opt.param_groups[0]['lr']}, step=step)
            losses = []
        if step in save_at:
            name = 'final.ckpt' if step == steps else f'step{step}.ckpt'
            save_checkpoint(os.path.join(out_dir, name), ema.model, {'spec': spec, 'step': step})
            log(f'saved {name}')
        if step % 2000 == 0 and step < steps:
            torch.save({'policy': policy.state_dict(), 'ema': ema.model.state_dict(), 'opt': opt.state_dict(),
                        'step': step, 'ema_step': ema.step_count, 'gen': gen.get_state()}, resume + '.tmp')
            os.replace(resume + '.tmp', resume)
    if os.path.exists(resume):
        os.remove(resume)
    if wb:
        wb.finish()


if __name__ == '__main__':
    main(sys.argv[1])
