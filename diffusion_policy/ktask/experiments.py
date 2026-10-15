"""
Stages of the K-task experiments (DESIGN.md). Each command writes job dirs and a manifest under ROOT/runs;
ktask.jobs submits them.

  python -m ktask.experiments base --seeds 1,2
  python -m ktask.experiments pilot --seed 0
  python -m ktask.experiments e1 --seeds 0,1,2 --chains 4
"""

import os

import click
import numpy as np

from ktask.data import SPLITS
from ktask.jobs import ROOT, Job, write_manifest

# 'v2': runs after the observation-history fix (2026-10-14 evening); v1 results (rollouts with observations
# 8 steps apart) stay in the unsuffixed directories
V = 'v2'
DATA = f'{ROOT}/data'
MIXED, EXPERT = f'{DATA}/mixed.npz', f'{DATA}/expert.npz'
RUNS = f'{ROOT}/runs'
EVAL_N = 128
BASE_STEPS = 40000
PPO_SPEC = {'n_envs': 1024, 'sigma': 0.02, 'noise_steps': [0, 10], 'lr': 3e-6, 'critic_lr': 1e-3, 'epochs': 5,
            'minibatch': 512, 'micro': 128, 'clip': 0.1, 'gamma': 0.99, 'lam': 0.95, 'target_kl': 0.05,
            'critic_epochs': 5, 'critic_warmup': 3}


def base_ckpt(s):
    return f'{RUNS}/base/s{s}/final.ckpt'


def wandb(group, name):
    return {'project': 'ktask', 'group': group, 'name': name}


def chain(seed, c):
    """A random order of the 16 RL tasks in which every prefix is balanced: the first 2 and first 4 have
    distinct objects and distinct targets, the first 8 have every object once and every target twice."""
    rng = np.random.RandomState(1000 * seed + c)
    R = SPLITS['R']
    by_obj = {o: [t for t in R if t // 4 == o] for o in range(8)}
    while True:
        first = [by_obj[o][rng.randint(2)] for o in range(8)]
        if sorted(np.bincount([t % 4 for t in first], minlength=4)) != [2, 2, 2, 2]:
            continue
        order = list(rng.permutation(first))
        if len({t % 4 for t in order[:4]}) == 4:
            rest = [t for t in R if t not in first]
            return [int(t) for t in order + list(rng.permutation(rest))]


@click.group()
def cli():
    pass


@cli.command()
@click.option('--seeds', default='0,1,2')
def base(seeds):
    """Base policies: BC on the mixed pool of R and H (24 tasks); evaluated on all 32 tasks."""
    jobs = []
    for s in map(int, seeds.split(',')):
        job = Job(f'base/s{s}', time='03:00:00')
        job.run('train', {'out_dir': job.dir, 'method': 'bc', 'data': [{'path': MIXED, 'tasks': 'RH'}],
                          'steps': BASE_STEPS, 'lr': 2e-4, 'warmup': 500, 'batch_size': 1024, 'seed': s,
                          'save_at': [10000, 20000, 30000], 'wandb': wandb('base', f'base_s{s}')})
        job.lines.append(f'python -m ktask.evaluate {job.dir}/final.ckpt -o {job.dir}/eval.json --n {EVAL_N}')
        jobs.append(job.write())
    write_manifest(f"base_s{seeds.replace(',', '')}", jobs)


def ppo_run(out_dir, init, tasks, seed, episodes, save_at, eval_tasks='all', **over):
    """A PPO spec, with its checkpoints evaluated on eval_tasks."""
    os.makedirs(out_dir, exist_ok=True)
    spec = {**PPO_SPEC, 'out_dir': out_dir, 'init_ckpt': init, 'tasks': [int(t) for t in tasks], 'seed': seed,
            'episodes': episodes, 'save_at': list(save_at), 'eval': {'tasks': eval_tasks, 'n': EVAL_N}, **over}
    path = f'{out_dir}/ppo.json'
    import json
    json.dump(spec, open(path, 'w'), indent=1)
    return path


def pack(name, runs, per_job=2, after=(), time='04:00:00'):
    """Jobs running `per_job` PPO runs side by side on one GPU; runs: list of (run dir, spec path)."""
    jobs = []
    for i in range(0, len(runs), per_job):
        job = Job(f'{name}/j{i // per_job}', after=after, time=time)
        job.parallel([('ppo', spec, f'{d}/stdout.txt') for d, spec in runs[i:i + per_job]])
        jobs.append(job.write())
    return jobs


@cli.command()
@click.option('--seed', default=1)
@click.option('--episodes', default=32768)
def pilot(seed, episodes):
    """PPO learning rate at 1,024 envs: K = 16 (all of R) at three rates, K = 1 at two, and K = 16 updating on
    3 random chain steps of the 10."""
    base = base_ckpt(seed)
    save_at = [episodes // 8, episodes // 4, episodes // 2, episodes]
    runs = []
    for k, lrs in ((16, (3e-6, 1e-5, 3e-5)), (1, (3e-6, 1e-5))):
        tasks = chain(seed, 0)[:k]
        for lr in lrs:
            d = f'{RUNS}/pilot/s{seed}/k{k}_lr{lr:.0e}'
            runs.append((d, ppo_run(d, base, tasks, seed, episodes, save_at, lr=lr)))
    d = f'{RUNS}/pilot/s{seed}/k16_lr1e-05_sub3'          # 3 of the 10 chain steps per update
    runs.append((d, ppo_run(d, base, chain(seed, 0), seed, episodes, save_at, lr=1e-5, chain_subsample=3)))
    jobs = pack(f'pilot/s{seed}', runs, per_job=1, after=[f'base-s{seed}'])
    write_manifest(f'pilot_s{seed}', jobs)


E1_KS = (1, 2, 4, 8)
LOG_KS = (1, 4)                 # runs whose episodes are logged for E2 (chains 0 and 1), plus K = 16
LR_PATH = (10000, [1000, 3000, 6000, 10000])   # offline arms: steps, evaluated checkpoints


def e1_dir(s, c, k):
    """K = 16 trains on all of R in every chain; its replicates are PPO seeds (c = 0, 1)."""
    if k == 16:
        return f'{RUNS}/e1{V}/s{s}/k16' if not c else f'{RUNS}/e1{V}/s{s}/k16_p{c}'
    return f'{RUNS}/e1{V}/s{s}/c{c}/k{k}'


@cli.command()
@click.option('--seeds', default='1,2,0')
@click.option('--chains', default=4)
@click.option('--episodes', default=32768)
@click.option('--lr', type=float, default=3e-6)
@click.option('--per_job', default=1)
def e1(seeds, chains, episodes, lr, per_job):
    """PPO on the first K tasks of each balanced chain (K = 1, 2, 4, 8) and on all 16 (two PPO seeds); the same
    total episodes for every K. Jobs are ordered chain by chain across seeds, so complete K-curves finish first."""
    save_at = [episodes // 8, episodes // 4, episodes // 2, episodes]
    seeds = [int(s) for s in seeds.split(',')]
    jobs = []
    for c in range(chains):
        for s in seeds:
            runs = []
            if c < 2:                                       # K = 16: all of R, PPO seeds s and 100 + s
                d = e1_dir(s, c, 16)
                runs.append((d, ppo_run(d, base_ckpt(s), chain(s, 0), s + 100 * c, episodes, save_at, lr=lr,
                                        log_rollouts=c == 0)))
            for k in E1_KS[::-1]:
                d = e1_dir(s, c, k)
                runs.append((d, ppo_run(d, base_ckpt(s), chain(s, c)[:k], s, episodes, save_at, lr=lr,
                                        log_rollouts=k in LOG_KS and c < 2)))
            jobs += pack(f'e1{V}/s{s}/c{c}', runs, per_job, after=[f'base-s{s}'], time='03:00:00')
    write_manifest(f'e1{V}', jobs)


def job_of(run_dir):
    """Name of the job (in any manifest under RUNS) that runs the PPO run in run_dir."""
    import glob
    import json
    for m in glob.glob(f'{RUNS}/*_manifest.json'):
        for j in json.load(open(m)):
            if os.path.exists(f"{j['dir']}/steps.sh") and f'{run_dir}/ppo.json' in open(f"{j['dir']}/steps.sh").read():
                return j['name']
    raise KeyError(run_dir)


def offline_job(name, spec, after, eval_tasks='all'):
    """Fine-tune from the base along LR_PATH, evaluating each saved checkpoint on eval_tasks."""
    job = Job(name, after=after, time='02:00:00')
    steps, evals = LR_PATH
    job.run('train', {**spec, 'out_dir': job.dir, 'steps': steps, 'warmup': 200, 'batch_size': 1024,
                      'save_at': evals})
    for n in evals:
        ck = f'{job.dir}/final.ckpt' if n == steps else f'{job.dir}/step{n}.ckpt'
        job.lines.append(f'python -m ktask.evaluate {ck} -o {job.dir}/eval_step{n}.json --tasks {eval_tasks} --n {EVAL_N}')
    return job.write()


@cli.command()
@click.option('--seeds', default='0,1,2')
@click.option('--episodes', default=32768)
@click.option('--lr', default=1e-6)
def e2(seeds, episodes, lr):
    """Same data, different update rule: filtered BC on each logged PPO run's successful episodes, and SFT
    on clean expert episodes of the same tasks, both from the base (lr fixed, evaluated along the way)."""
    jobs = []
    for s in map(int, seeds.split(',')):
        for c, k in [(None, 16)] + [(c, k) for k in LOG_KS for c in range(2)]:
            src = e1_dir(s, c, k)
            tag = 'k16' if k == 16 else f'c{c}/k{k}'
            tasks = SPLITS['R'] if k == 16 else chain(s, c)[:k]
            common = {'init_ckpt': base_ckpt(s), 'seed': s, 'lr': lr, 'method': 'bc'}
            jobs.append(offline_job(f'e2/s{s}/{tag}/filtered_bc', {**common, 'data': [
                {'glob': f'{src}/rollouts_*.npz', 'max_end': episodes, 'success_only': True}]},
                after=[job_of(src)], eval_tasks='all'))
            jobs.append(offline_job(f'e2/s{s}/{tag}/sft_expert', {**common, 'data': [
                {'path': EXPERT, 'tasks': [int(t) for t in tasks]}]}, after=[f'base-s{s}'], eval_tasks='all'))
    write_manifest(f"e2_s{seeds.replace(',', '')}", jobs)


@cli.command()
@click.option('--seeds', default='1,2,0')
@click.option('--chains', default=4)
@click.option('--n', default=512)
def e1x(seeds, chains, n):
    """Precise training-task evaluation for K = 1 runs (one task, so 128 episodes are noisy): the task with
    n episodes (seeds from 400000) for the base and each checkpoint, paired on start states."""
    jobs = []
    for s in map(int, seeds.split(',')):
        for c in range(chains):
            d = e1_dir(s, c, 1)
            task = chain(s, c)[0]
            job = Job(f'e1x/s{s}/c{c}', after=[job_of(d)], time='01:00:00')
            out = job.dir
            job.lines.append(f'python -m ktask.evaluate {base_ckpt(s)} -o {out}/base.json --tasks {task} --n {n} '
                             f'--slots {n} --seed 400000')
            for ep in (4096, 8192, 16384, 32768):
                job.lines.append(f'python -m ktask.evaluate {d}/ppo_ep{ep}.ckpt -o {out}/ep{ep}.json --tasks {task} '
                                 f'--n {n} --slots {n} --seed 400000')
            jobs.append(job.write())
    write_manifest('e1x', jobs)


LEARN_KS = (5, 20)
LEARN_STEPS = 2000


@cli.command()
@click.option('--seeds', default='0,1,2')
@click.option('--episodes', default=32768)
def e3(seeds, episodes):
    """Learnability of the 8 never-seen tasks (U): SFT on the first k clean demos of each U task (2000 steps,
    lr 1e-4, as robomimic's learn-B recipe) from the base and from PPO on K = 1, 4 (chain 0) and 16 tasks.
    Evaluated on U (learned) and H (kept)."""
    jobs = []
    for s in map(int, seeds.split(',')):
        starts = {'base': (base_ckpt(s), f'base-s{s}')}
        for k in (1, 4, 16):
            d = e1_dir(s, 0, k)
            starts[f'ppo_k{k}'] = (f'{d}/ppo_ep{episodes}.ckpt', job_of(d))
        for name, (ckpt, dep) in starts.items():
            job = Job(f'e3/s{s}/{name}', after=[dep], time='02:00:00')
            for k in LEARN_KS:
                out = f'{job.dir}/k{k}'
                os.makedirs(out, exist_ok=True)
                spec = {'out_dir': out, 'init_ckpt': ckpt, 'method': 'bc', 'seed': s, 'steps': LEARN_STEPS, 'lr': 1e-4,
                        'warmup': 100, 'batch_size': 1024, 'data': [{'path': EXPERT, 'tasks': 'U', 'first_n': k}]}
                import json
                json.dump(spec, open(f'{out}/train.json', 'w'), indent=1)
                job.lines.append(f'python -m ktask.train {out}/train.json')
                job.lines.append(f'python -m ktask.evaluate {out}/final.ckpt -o {out}/eval.json --tasks UH --n {EVAL_N}')
            jobs.append(job.write())
    write_manifest(f"e3_s{seeds.replace(',', '')}", jobs)


@cli.command()
@click.option('--seed', default=1)
@click.option('--episodes', default=32768)
def pilot2(seed, episodes):
    """After the first pilot (critic collapse): critic warmup and its own epochs; lr 1e-5 vs 3e-6 at 1,024
    envs for K = 16 and K = 1, lr 1e-5 at 256 envs (4x the iterations), and 3 of 10 chain steps per update."""
    base = base_ckpt(seed)
    save_at = [episodes // 8, episodes // 4, episodes // 2, episodes]
    runs = []
    for name, k, over in (('k16_lr1e-05', 16, {'lr': 1e-5}), ('k16_lr3e-06', 16, {'lr': 3e-6}),
                          ('k16_lr1e-05_n256', 16, {'lr': 1e-5, 'n_envs': 256, 'critic_warmup': 12}),
                          ('k1_lr1e-05', 1, {'lr': 1e-5}), ('k1_lr3e-06', 1, {'lr': 3e-6}),
                          ('k16_lr1e-05_sub3', 16, {'lr': 1e-5, 'chain_subsample': 3})):
        d = f'{RUNS}/pilot2/s{seed}/{name}'
        runs.append((d, ppo_run(d, base, chain(seed, 0)[:k], seed, episodes, save_at, **over)))
    jobs = pack(f'pilot2/s{seed}', runs, per_job=1, after=[f'base-s{seed}'])
    write_manifest(f'pilot2_s{seed}', jobs)


@cli.command()
@click.option('--seed', default=1)
@click.option('--episodes', default=12800)
def pilot3(seed, episodes):
    """The robomimic PPO regime (60 envs per iteration, lr 1e-6, minibatch 256, ~100+ iterations): 64 or 128
    envs per iteration, lr 1e-6 or 3e-6, K = 16 and K = 1; three runs share a GPU."""
    base = base_ckpt(seed)
    save_at = [episodes // 8, episodes // 4, episodes // 2, episodes]
    runs = []
    for name, k, over in (('k16_lr1e-06_n64', 16, {'lr': 1e-6, 'n_envs': 64}),
                          ('k1_lr1e-06_n64', 1, {'lr': 1e-6, 'n_envs': 64}),
                          ('k16_lr3e-06_n64', 16, {'lr': 3e-6, 'n_envs': 64}),
                          ('k1_lr3e-06_n64', 1, {'lr': 3e-6, 'n_envs': 64}),
                          ('k16_lr1e-06_n128', 16, {'lr': 1e-6, 'n_envs': 128}),
                          ('k1_lr1e-06_n128', 1, {'lr': 1e-6, 'n_envs': 128})):
        d = f'{RUNS}/pilot3/s{seed}/{name}'
        runs.append((d, ppo_run(d, base, chain(seed, 0)[:k], seed, episodes, save_at, minibatch=256, **over)))
    jobs = pack(f'pilot3/s{seed}', runs, per_job=3, after=[f'base-s{seed}'])
    write_manifest(f'pilot3_s{seed}', jobs)


# ------------------------------------------------------------------ follow-ups (2026-10-14 evening)

E2X_RUNS = ((0, 0), (1, 1), (2, 1))       # (seed, chain) of logged K = 1 runs: tall-air, puck-platform, bar-air


@cli.command()
@click.option('--episodes', default=32768)
@click.option('--n', default=16384, help='episodes to record per dataset')
def e2x(episodes, n):
    """Why filtered BC on PPO's episodes fails. For three K = 1 runs, BC (the E2 recipe, from the base) on:
    the final PPO policy's own episodes without chain noise (det) and with it (sigma), the base's own
    episodes (self), and PPO's logged episodes or clean expert demos with half of each batch replayed from
    the pretraining pool (replay)."""
    jobs = []
    for s, c in E2X_RUNS:
        src = e1_dir(s, c, 1)
        task = chain(s, c)[0]
        base = f'{RUNS}/e2x/s{s}/c{c}'
        col = Job(f'e2x/s{s}/c{c}/collect', after=[job_of(src)], time='02:00:00')
        data = {'det': f'{col.dir}/ppo_det.npz', 'sigma': f'{col.dir}/ppo_sigma.npz', 'self': f'{col.dir}/base_det.npz'}
        col.lines.append(f'python -m ktask.collect {src}/ppo_ep{episodes}.ckpt -o {data["det"]} --tasks {task} --n {n}')
        col.lines.append(f'python -m ktask.collect {src}/ppo_ep{episodes}.ckpt -o {data["sigma"]} --tasks {task} --n {n} '
                         f'--sigma 0.02')
        col.lines.append(f'python -m ktask.collect {base_ckpt(s)} -o {data["self"]} --tasks {task} --n {n}')
        jobs.append(col.write())
        common = {'init_ckpt': base_ckpt(s), 'seed': s, 'lr': 1e-6, 'method': 'bc'}
        replay = {'path': MIXED, 'tasks': 'RH', 'replay': True}
        arms = {f'fbc_{k}': [{'path': p, 'success_only': True}] for k, p in data.items()}
        arms['fbc_logged_replay'] = [{'glob': f'{src}/rollouts_*.npz', 'max_end': episodes, 'success_only': True}, replay]
        arms['sft_expert_replay'] = [{'path': EXPERT, 'tasks': [int(task)]}, replay]
        for name, d in arms.items():
            after = [col.name] if name.startswith('fbc_') and 'logged' not in name else [job_of(src)]
            jobs.append(offline_job(f'e2x/s{s}/c{c}/{name}', {**common, 'data': d}, after=after))
    write_manifest('e2x', jobs)


@cli.command()
@click.option('--episodes', default=32768)
def e3x(episodes):
    """Learnability curves on the never-seen tasks U: SFT on k in {1, 5} clean demos per U task at a gentle
    lr (2e-5), evaluated on U and H after 250, 1000 and 4000 steps, from the base, PPO on 1, 4 and 16 tasks
    (chain 0), and SFT on clean expert demos of the 16 R tasks (the supervised counterpart of PPO on 16)."""
    jobs = []
    for s in (0, 1, 2):
        starts = {'base': (base_ckpt(s), f'base-s{s}'),
                  'sft_expert_k16': (f'{RUNS}/e2/s{s}/k16/sft_expert/final.ckpt', f'e2-s{s}-k16-sft_expert')}
        for k in (1, 4, 16):
            d = e1_dir(s, 0, k)
            starts[f'ppo_k{k}'] = (f'{d}/ppo_ep{episodes}.ckpt', job_of(d))
        for name, (ckpt, dep) in starts.items():
            job = Job(f'e3x/s{s}/{name}', after=[dep], time='02:00:00')
            for kd in (1, 5):
                out = f'{job.dir}/k{kd}'
                os.makedirs(out, exist_ok=True)
                spec = {'out_dir': out, 'init_ckpt': ckpt, 'method': 'bc', 'seed': s, 'steps': 4000, 'lr': 2e-5,
                        'warmup': 100, 'batch_size': 1024, 'save_at': [250, 1000],
                        'data': [{'path': EXPERT, 'tasks': 'U', 'first_n': kd}]}
                import json
                json.dump(spec, open(f'{out}/train.json', 'w'), indent=1)
                job.lines.append(f'python -m ktask.train {out}/train.json')
                for st, ck in ((250, 'step250.ckpt'), (1000, 'step1000.ckpt'), (4000, 'final.ckpt')):
                    job.lines.append(f'python -m ktask.evaluate {out}/{ck} -o {out}/eval_step{st}.json --tasks UH '
                                     f'--n {EVAL_N}')
            jobs.append(job.write())
    write_manifest('e3x', jobs)


@cli.command()
@click.option('--episodes', default=32768)
@click.option('--rl_episodes', default=16384)
def e3r(episodes, rl_episodes):
    """RL learnability: PPO on the 8 never-seen tasks U (the E1 recipe) from the base, from PPO on the 16 R
    tasks, and from SFT on clean expert demos of the 16 R tasks."""
    save_at = [rl_episodes // 8, rl_episodes // 4, rl_episodes // 2, rl_episodes]
    runs, deps = [], []
    for s in (0, 1, 2):
        for name, ckpt, dep in (('base', base_ckpt(s), f'base-s{s}'),
                                ('ppo_k16', f'{e1_dir(s, 0, 16)}/ppo_ep{episodes}.ckpt', job_of(e1_dir(s, 0, 16))),
                                ('sft_expert_k16', f'{RUNS}/e2/s{s}/k16/sft_expert/final.ckpt', f'e2-s{s}-k16-sft_expert')):
            d = f'{RUNS}/e3r/s{s}/{name}'
            runs.append(((d, ppo_run(d, ckpt, SPLITS['U'], s, rl_episodes, save_at)), dep))
    jobs = []
    for (run, dep) in runs:
        jobs += pack(f'e3r/{os.path.basename(os.path.dirname(run[0]))}/{os.path.basename(run[0])}', [run], 1,
                     after=[dep], time='04:00:00')
    write_manifest('e3r', jobs)


@cli.command()
@click.option('--per_task', default=64, help='env slots per task per iteration')
@click.option('--iterations', default=32)
def e1p(per_task, iterations):
    """Equal data per task at equal iteration count: each iteration runs per_task episodes of each of the K
    tasks (n_envs = per_task * K, minibatch scaled the same way), for the same number of iterations as E1, so
    every K sees 2,048 episodes per task. K = 16 is E1's K = 16 (1,024 slots = 64 per task)."""
    jobs = []
    for s in (1, 2, 0):
        for c in (0, 1):
            runs = []
            for k in (8, 4, 2, 1):
                n_envs = per_task * k
                eps = n_envs * iterations
                d = f'{RUNS}/e1p/s{s}/c{c}/k{k}'
                runs.append((d, ppo_run(d, base_ckpt(s), chain(s, c)[:k], s, eps,
                                        [n_envs * i for i in (iterations // 8, iterations // 4, iterations // 2, iterations)],
                                        n_envs=n_envs, minibatch=max(32, 32 * k))))
            jobs += pack(f'e1p/s{s}/c{c}', runs, per_job=1, after=[f'base-s{s}'], time='03:00:00')
    write_manifest('e1p', jobs)


@cli.command()
@click.option('--seeds', default='0,1')
@click.option('--episodes', default=32768)
def pilotv2(seeds, episodes):
    """After the observation-history fix the bases are stronger (~0.70 on all tasks): does PPO still improve
    the training tasks? K = 16 and K = 1 (chain 0) at lr 3e-6 (E1's) and 1e-5."""
    save_at = [episodes // 8, episodes // 4, episodes // 2, episodes]
    jobs = []
    for s in map(int, seeds.split(',')):
        runs = []
        for k in (16, 1):
            for lr in (3e-6, 1e-5):
                d = f'{RUNS}/pilotv2/s{s}/k{k}_lr{lr:.0e}'
                runs.append((d, ppo_run(d, base_ckpt(s), chain(s, 0)[:k], s, episodes, save_at, lr=lr)))
        jobs += pack(f'pilotv2/s{s}', runs, per_job=1, after=[f'base-s{s}'], time='03:00:00')
    write_manifest('pilotv2', jobs)


if __name__ == '__main__':
    cli()
