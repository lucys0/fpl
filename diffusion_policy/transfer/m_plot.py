"""
Figures for the redesigned transfer experiment (mean +/- SE over seeds).

  left:  change on task B against change on task A, per arm (post-training on A only)
  right: task B success after fine-tuning on K clean B demos, against K

  python -m transfer.m_plot [--seeds 0,1,2] [--out results/m_transfer.png]
"""

import json
import os

import click
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

R = '/iris/u/lucyshi/fpl_transfer/runs'
KS = (3, 5, 10)
LEARN_B_K = 10
COLORS = {'sft_pool': 'tab:blue', 'filtered_bc': 'tab:orange', 'awr': 'tab:green', 'udrl_scalar': 'tab:purple',
          'udrl_axes': 'tab:pink', 'sft_expert': 'tab:gray', 'ppo': 'tab:red'}
LABELS = {'sft_pool': 'SFT (pool)', 'filtered_bc': 'filtered BC', 'awr': 'AWR', 'udrl_scalar': 'UDRL (scalar)',
          'udrl_axes': 'UDRL (axes)', 'sft_expert': 'SFT (expert demos)', 'ppo': 'PPO'}


def res(p):
    return json.load(open(p))['results'] if os.path.exists(p) else None


def succ(r, peg):
    return r[peg]['success'] if r and r.get(peg) else None


def mean_se(v):
    v = [x for x in v if x is not None]
    if not v:
        return None, None
    return float(np.mean(v)), (float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0)


def learn(job_dir, m3_dir, tag, k):
    name = f'{tag}_learnB' + ('' if k == LEARN_B_K else f'_k{k}') + '_eval.json'
    return res(f'{job_dir}/{name}') or res(f'{m3_dir}/{name}')


def arms(seeds):
    """(name, method, stage, budget, {seed: eval}, {seed: job dir}, {seed: m3 dir}, tag)"""
    out = []
    for stage in ('m1r', 'm1'):
        for m in COLORS:
            if m == 'ppo':
                continue
            for n in (200, 1000):
                out.append((f'{stage}/{m}/n{n}', m, stage, n,
                            {s: res(f'{R}/{stage}/s{s}/{m}/n{n}_eval.json') for s in seeds},
                            {s: f'{R}/{stage}/s{s}/{m}' for s in seeds},
                            {s: f'{R}/m3/{stage}/s{s}/{m}' for s in seeds}, f'n{n}'))
    for n in (200, 1000, 3000, 6000):
        out.append((f'm2/ppo/n{n}', 'ppo', 'm2', n, {s: res(f'{R}/m2/s{s}/ppo/n{n}_eval.json') for s in seeds},
                    {s: f'{R}/m2/s{s}/ppo' for s in seeds}, {s: f'{R}/m3/m2/s{s}/ppo' for s in seeds}, f'n{n}'))
    for m in ('filtered_bc', 'awr'):
        for n in (200, 1000, 3000):
            out.append((f'm4/{m}/n{n}', m, 'm4', n, {s: res(f'{R}/m4/s{s}/{m}/n{n}_eval.json') for s in seeds},
                        {s: f'{R}/m4/s{s}/{m}' for s in seeds}, {s: f'{R}/m3/m4/s{s}/{m}' for s in seeds}, f'n{n}'))
    return out


@click.command()
@click.option('--seeds', default='0,1,2')
@click.option('--out', default='/iris/u/lucyshi/fpl_transfer/results/m_transfer.png')
def main(seeds, out):
    seeds = [int(s) for s in seeds.split(',')]
    base = {s: res(f'{R}/m0/base_s{s}/eval.json') for s in seeds}
    base_learn = {k: {s: learn(f'{R}/m0/base_s{s}', f'{R}/m3/m0/s{s}/base', 'base', k) for s in seeds} for k in KS}
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))

    # ---- B change vs A change
    ppo_pts, online = [], {}
    for name, m, stage, n, ev, _, _, _ in arms(seeds):
        dA = mean_se([succ(ev[s], 'left') - succ(base[s], 'left') for s in seeds if ev[s] and base[s]])
        dB = mean_se([succ(ev[s], 'right') - succ(base[s], 'right') for s in seeds if ev[s] and base[s]])
        if dA[0] is None:
            continue
        if m == 'ppo':
            ppo_pts.append((n, dA, dB))
            continue
        if stage == 'm4':
            online.setdefault(m, []).append((n, dA, dB))
            continue
        filled = stage == 'm1r'
        ax1.errorbar(dA[0], dB[0], xerr=dA[1], yerr=dB[1], fmt='o' if n == 1000 else 's', ms=8 if n == 1000 else 6,
                     color=COLORS[m], mfc=COLORS[m] if filled else 'white', capsize=2, alpha=0.9)
    if ppo_pts:
        ppo_pts.sort()
        ax1.plot([0] + [p[1][0] for p in ppo_pts], [0] + [p[2][0] for p in ppo_pts], '-', color=COLORS['ppo'], alpha=0.5)
        for n, dA, dB in ppo_pts:
            ax1.errorbar(dA[0], dB[0], xerr=dA[1], yerr=dB[1], fmt='^', ms=8, color=COLORS['ppo'], capsize=2)
            ax1.annotate(str(n), (dA[0], dB[0]), textcoords='offset points', xytext=(5, 4), fontsize=8,
                         color=COLORS['ppo'])
    for m, pts in online.items():
        pts.sort()
        ax1.plot([0] + [p[1][0] for p in pts], [0] + [p[2][0] for p in pts], '--', color=COLORS[m], alpha=0.5)
        for n, dA, dB in pts:
            ax1.errorbar(dA[0], dB[0], xerr=dA[1], yerr=dB[1], fmt='D', ms=7, color=COLORS[m], mfc='white', capsize=2)
            ax1.annotate(str(n), (dA[0], dB[0]), textcoords='offset points', xytext=(5, -10), fontsize=8, color=COLORS[m])
    (x0, x1), (y0, y1) = ax1.get_xlim(), ax1.get_ylim()
    lo, hi = min(x0, y0), max(x1, y1)
    ax1.plot([lo, hi], [lo, hi], ':', color='k', lw=0.8, label='ΔB = ΔA')
    ax1.set_xlim(x0, x1); ax1.set_ylim(y0, y1)
    ax1.axhline(0, color='k', lw=0.6); ax1.axvline(0, color='k', lw=0.6)
    ax1.set_xlabel('Δ success on A (post-trained task)'); ax1.set_ylabel('Δ success on B (zero-shot)')
    ax1.set_title('Transfer: B change vs A change\nfilled = replay, hollow = no replay; ○ N=1000, □ N=200; '
                  '▲ PPO, ◇ iterated online (episodes)')
    handles = [plt.Line2D([], [], marker='o', ls='', color=c, label=LABELS[m]) for m, c in COLORS.items()]
    ax1.legend(handles=handles, fontsize=8, loc='best')

    # ---- B learnability curves (with replay, N=1000; PPO at its largest budget with data)
    for ax, stage, title in ((ax2, 'm1r', 'with replay'), (ax3, 'm1', 'no replay')):
        bm = [mean_se([succ(base_learn[k][s], 'right') for s in seeds]) for k in KS]
        if bm[0][0] is not None or bm[-1][0] is not None:
            ax.errorbar(KS, [b[0] if b[0] is not None else np.nan for b in bm],
                        yerr=[b[1] or 0 for b in bm], fmt='-o', color='k', lw=2, label='base policy', capsize=2)
        for name, m, st, n, ev, jd, m3d, tag in arms(seeds):
            if not ((st == stage and n == 1000) or (stage == 'm1' and m == 'ppo' and n == 6000)
                    or (stage == 'm1' and st == 'm4' and n == 3000)):
                continue
            pts = [mean_se([succ(learn(jd[s], m3d[s], tag, k), 'right') for s in seeds]) for k in KS]
            if all(p[0] is None for p in pts):
                continue
            ax.errorbar(KS, [p[0] if p[0] is not None else np.nan for p in pts], yerr=[p[1] or 0 for p in pts],
                        fmt='-^' if m == 'ppo' else ('--D' if st == 'm4' else '-o'), color=COLORS[m], capsize=2,
                        mfc='white' if st == 'm4' else COLORS[m], alpha=0.85,
                        label=LABELS[m] + (' (6000 ep.)' if m == 'ppo' else
                                           ' (online, 3000 ep.)' if st == 'm4' else ' (N=1000)'))
        ax.set_xscale('log'); ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel('clean B demos for fine-tuning (K)'); ax.set_ylabel('B success after fine-tuning')
        ax.set_title(f'Learning B after post-training on A ({title})' + ('\nPPO and online arms have no replay'
                                                                         if stage == 'm1' else ''))
        ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
