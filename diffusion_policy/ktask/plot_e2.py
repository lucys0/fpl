"""
E2 figure: the same tasks, different update rule. For K = 1, 4, 16: change on the training tasks vs change on
held-out tasks (H and the rest of R) along PPO (episodes), filtered BC on PPO's own successful episodes and
SFT on clean expert demos (gradient steps), each from the base; mean over runs, 95% CI over runs.

  python -m ktask.plot_e2 [--out results/e2.png]
"""

import click
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ktask.analyze import E2_ARMS, e2_points

COLORS = {'ppo': 'tab:red', 'filtered_bc': 'tab:orange', 'sft_expert': 'tab:gray'}


def ci(v):
    v = np.asarray(v)
    return 1.96 * v.std() / np.sqrt(len(v)) if len(v) > 1 else 0.0


@click.command()
@click.option('--out', default='/iris/u/lucyshi/fpl_transfer/ktask/results/e2.png')
def main(out):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
    for ax, k in zip(axes, (1, 4, 16)):
        pts = e2_points(k)
        for arm, label, points, _ in E2_ARMS:
            xs, ys = [0.0], [0.0]
            for i, p in enumerate(points):
                rs = pts.get(arm, {}).get(p)
                if not rs:
                    continue
                x, y = [r['train'] for r in rs], [r['held'] for r in rs]
                xs.append(np.mean(x))
                ys.append(np.mean(y))
                ax.errorbar(np.mean(x), np.mean(y), xerr=ci(x), yerr=ci(y), fmt='o', color=COLORS[arm],
                            ms=4 + 1.5 * i, capsize=2, label=f'{label} (n={len(rs)})' if i == len(points) - 1 else None)
            ax.plot(xs, ys, '-', color=COLORS[arm], alpha=0.6)
        ax.axhline(0, color='k', lw=0.6)
        ax.axvline(0, color='k', lw=0.6)
        ax.plot([-0.05, 0.4], [-0.05, 0.4], color='k', lw=0.5, alpha=0.3)
        ax.set_xlabel('change on the K training tasks')
        ax.set_ylabel('change on held-out tasks (H + rest of R)')
        ax.set_title(f'K = {k} (PPO: 4k/8k/16k/32k episodes; offline: 1k/3k/6k/10k steps)', fontsize=9.5)
        ax.legend(fontsize=7.5, loc='lower right')
    fig.suptitle('Same tasks, different update rule (each from the base; marker size grows along training)', fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
