"""
Summary figure for the third round (mean over seeds; error bars: 95% paired-bootstrap CIs where a change is shown).

  (a, b) same data, different update rule: change on A vs change on B along training, PPO vs filtered BC / AWR
         trained on exactly PPO's logged episodes (first pair, second pair)
  (c, d) B never in pretraining: B success after SFT on k clean B demos, from each A-pretrained start
  (e)    round-nut success with the round peg moved along PPO's release shift
  (f)    PPO on B: B success along PPO training, second pair (round nut) and A-only starts (after 5 demos)

  python -m transfer.round3_plot [--out results/round3.png]
"""

import click
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from transfer.round3 import R, m_base, stat, x_base


def delta(paths, task, ref):
    r = stat(paths, task, ref)
    return (np.nan, 0, 0) if r is None else (r['d'], r['d'] - r['lo'], r['hi'] - r['d'])


def mean(paths, task):
    r = stat(paths, task)
    return np.nan if r is None else r['mean']


def path_panel(ax, stage, B, base, title):
    arms = [('ppo', 'PPO (episodes logged)', 'tab:red', [1000, 3000, 6000], lambda n: f'eval_n{n}'),
            ('filtered_bc', 'filtered BC on PPO\'s episodes', 'tab:orange', [1000, 3000, 6000, 10000], lambda n: f'eval_step{n}'),
            ('awr', 'AWR on PPO\'s episodes', 'tab:green', [1000, 3000, 6000, 10000], lambda n: f'eval_step{n}')]
    for arm, label, c, points, sub in arms:
        xs, ys = [0.0], [0.0]
        for i, n in enumerate(points):
            p = lambda s, n=n: f'{R}/{stage}/s{s}/{arm}/{sub(n)}/eval.json'
            (a, ae0, ae1), (b, be0, be1) = delta(p, 'left', base), delta(p, B, base)
            xs.append(a); ys.append(b)
            ax.errorbar(a, b, xerr=[[ae0], [ae1]], yerr=[[be0], [be1]], fmt='o', color=c, ms=4 + 1.5 * i, capsize=2,
                        label=label if i == 0 else None)
        ax.plot(xs, ys, '-', color=c, alpha=0.5)
    ax.axhline(0, color='k', lw=0.6); ax.axvline(0, color='k', lw=0.6)
    lim = [-0.05, 0.35]
    ax.plot(lim, lim, color='k', lw=0.6, alpha=0.4)
    ax.set_xlim(*lim)
    ax.set_xlabel('change in A success vs base'); ax.set_ylabel(f'change in B success vs base')
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7.5, loc='lower left')


@click.command()
@click.option('--out', default='/iris/u/lucyshi/fpl_transfer/results/round3.png')
def main(out):
    fig, axes = plt.subplots(2, 3, figsize=(19, 11.5))
    path_panel(axes[0, 0], 'mlog', 'right', m_base,
               '(a) Same data, different update rule: first pair\n(points: 1000/3000/6000 PPO episodes; '
               '1000/3000/6000/10000 offline steps)')
    path_panel(axes[0, 1], 'xlog', 'round', x_base,
               '(b) Same data, different update rule: second pair (B = round nut)')

    starts = [('base', 'A-only base (BC on A)', 'k'), ('ppoA6000', 'then PPO on A', 'tab:red'),
              ('sft_expert', 'then SFT on clean A demos', 'tab:gray'), ('filtered_bc', 'then filtered BC on A', 'tab:orange')]
    ks = [5, 10, 25, 50, 100]
    for ax, task, name in ((axes[0, 2], 'right', 'B1 = right peg'), (axes[1, 0], 'round', 'B2 = round nut')):
        for key, label, c in starts:
            ys = [mean(lambda s, k=k: f'{R}/a2/s{s}/{key}/{task}/eval_k{k}/eval.json', task) for k in ks]
            ax.plot(ks, ys, '-o', color=c, label=label, ms=4)
        ax.set_xscale('log'); ax.set_xticks(ks); ax.set_xticklabels(ks)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel('clean B demos (SFT, 2000 steps)'); ax.set_ylabel('B success')
        ax.set_title(f'({"c" if task == "right" else "d"}) B never in pretraining: learning {name} from demos\n'
                     f'(zero-shot B = 0 for every start)', fontsize=10)
        ax.legend(fontsize=7.5, loc='lower right')

    ax = axes[1, 1]
    ts = [-0.425, 0.0, 0.425, 0.85, 1.275, 1.7]
    zero = {'base': x_base, 'ppoA6000': lambda s: f'{R}/x2/s{s}/ppo/n6000_eval.json',
            'sft_expert_r1000': lambda s: f'{R}/x1r/s{s}/sft_expert/n1000_eval.json'}
    for arm, label, c in (('base', 'base', 'k'), ('ppoA6000', 'PPO on A (6000 ep.)', 'tab:red'),
                          ('sft_expert_r1000', 'SFT expert, replay', 'tab:gray')):
        ys = [mean(zero[arm] if t == 0 else (lambda s, t=t: f'{R}/xpeg/s{s}/{arm}/t{t:+.3f}/eval.json'), 'round')
              for t in ts]
        ax.plot(ts, ys, '-o', color=c, label=label, ms=4)
    ax.axvline(0, color='k', lw=0.6, ls=':')
    ax.text(0.98, 0.04, "PPO's curve = the base's curve\nshifted right by 0.4 cm, about 0.1 lower",
            color='tab:red', fontsize=8.5, transform=ax.transAxes, ha='right', va='bottom')
    ax.set_xlabel('round peg moved along PPO\'s release shift (cm; 0 = original)'); ax.set_ylabel('round-nut success')
    ax.set_title('(e) Second pair: moving the round peg\n(half of PPO\'s round-nut loss is its shifted release)', fontsize=10)
    ax.legend(fontsize=7.5, loc='upper right')

    ax = axes[1, 2]
    ns = [0, 200, 1000, 3000]
    for key, label, c, zero_fn in (('base', 'second pair: from base', 'k', x_base),
                                   ('ppoA6000', 'second pair: from PPO on A', 'tab:red',
                                    lambda s: f'{R}/x2/s{s}/ppo/n6000_eval.json'),
                                   ('sft_expert_r1000', 'second pair: from SFT expert, replay', 'tab:gray',
                                    lambda s: f'{R}/x1r/s{s}/sft_expert/n1000_eval.json')):
        ys = [mean(zero_fn, 'round')] + [mean(lambda s, n=n: f'{R}/xppob/s{s}/{key}/eval_n{n}/eval.json', 'round')
                                         for n in ns[1:]]
        ax.plot(ns, ys, '-o', color=c, label=label, ms=4)
    for key, label, c in (('base', 'B never in pretraining: from A-only base', 'k'),
                          ('ppoA6000', 'B never in pretraining: from PPO on A', 'tab:red')):
        ys = [mean(lambda s: f'{R}/a2/s{s}/{key}/right/eval_k5/eval.json', 'right')] + \
             [mean(lambda s, n=n: f'{R}/a3/s{s}/{key}/right_k5/eval_n{n}/eval.json', 'right') for n in ns[1:]]
        ax.plot(ns, ys, '--s', color=c, mfc='white', label=label + ' (right peg, after 5 demos)', ms=4)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel('PPO episodes on B'); ax.set_ylabel('B success')
    ax.set_title('(f) PPO on B: RL on A does not speed it up', fontsize=10)
    ax.legend(fontsize=7, loc='lower right')
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
