"""
Writes job directories for the transfer experiments. Each job directory holds
its train specs and a steps.sh whose steps are skipped when their outputs
already exist, so a requeued job picks up where it stopped.

  python -m transfer.experiments e0-calib            # base policies, K demos per peg (narrow nut range)
  python -m transfer.experiments e0-wide             # evaluate them on the wide nut range
  python -m transfer.experiments e1 --base_k 20      # post-train on A, eval A/B, adapt to B (wide range)
  python -m transfer.experiments e2 --base_k 20      # online: PPO vs filtered BC / AWR in rounds

Task A = left peg, task B = right peg. Base policies learn from demos with the
nut starting in the original narrow range; everything after pretraining
(post-training data, adaptation, evaluation) uses a wider range, where those
policies succeed about half the time. Grasping from unfamiliar positions is
shared between the two tasks; transport and insertion are not.
"""

import json
import math
import os

import click

ROOT = '/iris/u/lucyshi/fpl_transfer'
DATA = f'{ROOT}/data'
RUNS = f'{ROOT}/runs'
ENV_META = f'{DATA}/pretrain_left.hdf5'
EVAL_SEED = 300000          # held-out nut positions for every evaluation
WIDE = '[[-0.3,-0.3],[0.1,0.3]]'   # nut start range after pretraining (pretraining: [[-0.2,-0.2],[0,0.2]])
EVAL_N = 100                # episodes per peg per evaluation
WANDB_PROJECT = 'fpl_transfer'

BASE_STEPS = 20000
POST_STEPS = 5000           # post-training on A: same number of gradient steps for every method
ADAPT_STEPS = 3000          # fixed recipe for learning B afterwards: filtered BC on B rollouts
ADAPT_M = 200               # B rollouts collected for that
BUDGETS = (200, 1000)       # episodes of task A per method
METHODS = ('sft', 'filtered_bc', 'awr', 'udrl_scalar', 'udrl_axes')

ONLINE_BUDGETS = (200, 1000, 3000)   # episodes of task A collected online (iterated methods)
PPO_BUDGETS = (200, 1000, 3000, 6000)  # PPO is evaluated further out; 1000 episodes did not move it
ROUND_EPISODES = 200                 # per round of the iterated supervised methods
ROUND_STEPS = 2000                   # gradient steps per round
# Noise must cover every integration step: the network is shared across steps, and with noise on
# steps 0-5 only, updates changed the deterministic steps outside the PPO ratio and the policy collapsed.
# Small sigma keeps the base policy intact (0.02 on all steps: 0.40 success vs 0.50 deterministic) but
# makes ratios sensitive, hence the small learning rate.
PPO_SPEC = {'sigma': 0.02, 'noise_steps': [0, 10], 'lr': 1e-6, 'critic_lr': 1e-3, 'epochs': 5,
            'minibatch': 256, 'clip': 0.1, 'gamma': 0.99, 'lam': 0.95, 'target_kl': 0.05, 'n_envs': 60}


class Job:
    def __init__(self, name, after=(), kind='gpu', cpus=None):
        """kind 'cpu': evaluation only, run without a GPU (transfer.submit bundles these); cpus: per-job
        CPU request (default: transfer.submit's)."""
        self.dir = f'{RUNS}/{name}'
        self.name = name.replace('/', '-')
        self.after = list(after)
        self.kind, self.cpus = kind, cpus
        self.lines = []
        os.makedirs(self.dir, exist_ok=True)

    def train(self, tag, spec):
        spec = {**spec, 'out_dir': f'{self.dir}/{tag}',
                'wandb': {'project': WANDB_PROJECT, 'group': self.name.split('-')[0], 'name': f'{self.name}-{tag}'}}
        path = f'{self.dir}/{tag}.json'
        json.dump(spec, open(path, 'w'), indent=1)
        self.lines.append(f'[ -f {self.dir}/{tag}/final.ckpt ] || python -m transfer.train {path}')
        return f'{self.dir}/{tag}/final.ckpt'

    def rollout(self, tag, ckpt, pegs, n, seed_start, record=False, nut_range=WIDE):
        out = f'{self.dir}/{tag}.json'
        rec = f' --record {self.dir}/{tag}.hdf5' if record else ''
        rng = f" --nut_range '{nut_range}'" if nut_range else ''
        self.lines.append(f'[ -f {out} ] || python -m transfer.rollout --ckpt {ckpt} --pegs {pegs} --n {n} '
                          f'--seed_start {seed_start} --env_meta_from {ENV_META} --out_json {out}{rec}{rng}')
        return f'{self.dir}/{tag}.hdf5'

    def write(self):
        with open(f'{self.dir}/steps.sh', 'w') as f:
            f.write('#!/bin/bash\nset -eo pipefail\n' + '\n'.join(self.lines) + f'\ntouch {self.dir}/DONE\n')
        out = {'name': self.name, 'dir': self.dir, 'after': self.after, 'kind': self.kind}
        return {**out, 'cpus': self.cpus} if self.cpus else out


def write_manifest(stage, jobs):
    path = f'{RUNS}/{stage}_manifest.json'
    json.dump(jobs, open(path, 'w'), indent=1)
    print(f"{len(jobs)} jobs -> {path}")


def base_spec(k, seed):
    return {'method': 'bc', 'seed': seed, 'steps': BASE_STEPS, 'lr': 2e-4, 'warmup': 500, 'batch_size': 1024,
            'data': [{'path': f'{DATA}/pretrain_left.hdf5', 'max_per_peg': k, 'rng_seed': seed},
                     {'path': f'{DATA}/pretrain_right.hdf5', 'max_per_peg': k, 'rng_seed': seed}]}


def add_adapt_b(job, tag, ckpt, seed):
    """Fixed recipe for 'how fast is B learned from here': filtered BC on M B rollouts, then eval B."""
    rollouts = job.rollout(f'{tag}_brollouts', ckpt, 'right', ADAPT_M, 2_000_000 + 10_000 * seed, record=True)
    adapted = job.train(f'{tag}_adaptB', {'method': 'bc', 'init_ckpt': ckpt, 'seed': seed, 'steps': ADAPT_STEPS,
                                          'lr': 1e-4, 'warmup': 200, 'batch_size': 1024,
                                          'data': [{'path': rollouts, 'success_only': True}]})
    job.rollout(f'{tag}_adaptB_eval', adapted, 'right', EVAL_N, EVAL_SEED)


@click.group()
def cli():
    pass


@cli.command()
@click.option('--ks', default='10,20,40,80')
@click.option('--seeds', default='0,1,2')
def e0_calib(ks, seeds):
    """Base policies trained on K demos per peg; evaluated on both pegs."""
    jobs = []
    for k in map(int, ks.split(',')):
        for s in map(int, seeds.split(',')):
            job = Job(f'e0/base_k{k}_s{s}')
            ckpt = job.train('base', base_spec(k, s))
            job.rollout('eval', ckpt, 'left,right', EVAL_N, EVAL_SEED, nut_range=None)
            jobs.append(job.write())
    write_manifest('e0', jobs)


@cli.command()
@click.option('--ks', default='10,20,40,80')
@click.option('--seeds', default='0,1,2')
def e0_wide(ks, seeds):
    """Evaluate the calibration base policies on the wide nut range (one job per K)."""
    jobs = []
    for k in map(int, ks.split(',')):
        job = Job(f'e0w/k{k}')
        for s in map(int, seeds.split(',')):
            job.rollout(f'eval_s{s}', f'{RUNS}/e0/base_k{k}_s{s}/base/final.ckpt', 'left,right', EVAL_N, EVAL_SEED)
        jobs.append(job.write())
    write_manifest('e0w', jobs)


@cli.command()
@click.option('--base_k', required=True, type=int)
@click.option('--seeds', default='0,1,2')
@click.option('--methods', default=','.join(METHODS))
@click.option('--replay', is_flag=True, help='Mix the base policy\'s pretraining demos into post-training (50% of steps)')
def e1(base_k, seeds, methods, replay):
    """Post-train each base policy on task A only, then evaluate A, B, and how fast B is learned.

    One job per (seed, method) runs both budgets. The base policy's A rollouts
    (what filtered BC / AWR / UDRL train on) are collected once per seed by a
    small job the others wait for; smaller budgets are prefixes of larger ones.
    """
    jobs = []
    stage = 'e1r' if replay else 'e1'
    for s in map(int, seeds.split(',')):
        base = f'{RUNS}/e0/base_k{base_k}_s{s}/base/final.ckpt'
        replay_data = ([{**d, 'replay': True} for d in base_spec(base_k, s)['data']] if replay else [])
        if replay:
            # the replay variant reuses the no-replay stage's base rollouts, evals, adaptation and ceiling
            arollouts = f'{RUNS}/e1/k{base_k}_s{s}/arollouts/arollouts.hdf5'
            rollouts_job = f'e1-k{base_k}_s{s}-arollouts'
            prefix = f'{stage}/k{base_k}_s{s}'
            for m in methods.split(','):
                job = Job(f'{prefix}/{m}', after=[] if m == 'sft' else [rollouts_job])
                for n in BUDGETS:
                    common = {'init_ckpt': base, 'seed': s, 'steps': POST_STEPS, 'lr': 1e-4, 'warmup': 200,
                              'batch_size': 1024, 'replay_frac': 0.5}
                    job_spec = method_spec(m, n, arollouts, common)
                    job_spec['data'] = job_spec['data'] + replay_data
                    ckpt = job.train(f'n{n}', job_spec)
                    job.rollout(f'n{n}_eval', ckpt, 'left,right', EVAL_N, EVAL_SEED)
                    if not m.startswith('udrl'):
                        add_adapt_b(job, f'n{n}', ckpt, s)
                jobs.append(job.write())
            continue
        prefix = f'{stage}/k{base_k}_s{s}'
        job = Job(f'{prefix}/arollouts')
        job.rollout('base_eval', base, 'left,right', EVAL_N, EVAL_SEED)
        arollouts = job.rollout('arollouts', base, 'left', max(BUDGETS), 1_000_000 + 10_000 * s, record=True)
        jobs.append(job.write())
        rollouts_job = job.name

        # base policy: how fast B is learned from it, and the ceiling (B demos directly)
        job = Job(f'{prefix}/base')
        add_adapt_b(job, 'base', base, s)
        ceil = job.train('ceilB', {'method': 'bc', 'init_ckpt': base, 'seed': s, 'steps': POST_STEPS, 'lr': 1e-4,
                                   'warmup': 200, 'batch_size': 1024,
                                   'data': [{'path': f'{DATA}/ceil_right_wide.hdf5', 'first_n': max(BUDGETS)}]})
        job.rollout('ceilB_eval', ceil, 'left,right', EVAL_N, EVAL_SEED)
        jobs.append(job.write())

        for m in methods.split(','):
            job = Job(f'{prefix}/{m}', after=[] if m == 'sft' else [rollouts_job])
            for n in BUDGETS:
                common = {'init_ckpt': base, 'seed': s, 'steps': POST_STEPS, 'lr': 1e-4, 'warmup': 200, 'batch_size': 1024}
                ckpt = job.train(f'n{n}', method_spec(m, n, arollouts, common))
                job.rollout(f'n{n}_eval', ckpt, 'left,right', EVAL_N, EVAL_SEED)
                if not m.startswith('udrl'):      # adapting a return-conditioned policy needs its own recipe
                    add_adapt_b(job, f'n{n}', ckpt, s)
            jobs.append(job.write())
    # rollout-collection jobs gate the others, so they go first (and get iris-hi slots)
    jobs.sort(key=lambda j: not j['name'].endswith('arollouts'))
    write_manifest(f"{stage}_k{base_k}_s{seeds.replace(',', '')}", jobs)


def method_spec(m, n, arollouts, common):
    """Post-training spec on task A for method m with a budget of n episodes."""
    if m == 'sft':
        return {**common, 'method': 'bc', 'data': [{'path': f'{DATA}/sft_left_wide.hdf5', 'first_n': n}]}
    if m == 'filtered_bc':
        return {**common, 'method': 'bc', 'data': [{'path': arollouts, 'first_n': n, 'success_only': True}]}
    if m == 'awr':
        return {**common, 'method': 'awr', 'data': [{'path': arollouts, 'first_n': n}]}
    return {**common, 'method': 'udrl', 'udrl_mode': m.split('_')[1], 'data': [{'path': arollouts, 'first_n': n}]}


@cli.command()
@click.option('--base_k', required=True, type=int)
@click.option('--seeds', default='0,1,2')
@click.option('--methods', default='ppo,filtered_bc_iter,awr_iter')
@click.option('--ppo_spec', default='{}', help='JSON overrides of PPO_SPEC')
@click.option('--tag', default='', help='Suffix for the job directories (e.g. for a PPO sweep)')
@click.option('--replay', is_flag=True, help='Iterated methods also train on the pretraining demos (50% of steps)')
def e2(base_k, seeds, methods, ppo_spec, tag, replay):
    """Online post-training on A with matched episode budgets: PPO vs filtered BC / AWR in rounds.

    The iterated methods collect ROUND_EPISODES with the current policy, fine-tune it on
    everything collected so far (successes only for filtered BC), and repeat. Every
    checkpoint at an ONLINE_BUDGETS episode count is evaluated on A and B and adapted to B.
    """
    jobs = []
    stage = 'e2r' if replay else 'e2'
    for s in map(int, seeds.split(',')):
        base = f'{RUNS}/e0/base_k{base_k}_s{s}/base/final.ckpt'
        replay_data = [{**d, 'replay': True} for d in base_spec(base_k, s)['data']] if replay else []
        for m in methods.split(','):
            job = Job(f'{stage}/k{base_k}_s{s}/{m}{tag}')
            if m == 'ppo':
                spec = {**PPO_SPEC, **json.loads(ppo_spec), 'init_ckpt': base, 'peg': 'left', 'seed': s,
                        'episodes': max(PPO_BUDGETS), 'save_at': list(PPO_BUDGETS), 'env_meta_from': ENV_META,
                        'nut_range': json.loads(WIDE), 'out_dir': f'{job.dir}/ppo'}
                json.dump(spec, open(f'{job.dir}/ppo.json', 'w'), indent=1)
                job.lines.append(f'[ -f {job.dir}/ppo/ppo_ep{max(PPO_BUDGETS)}.ckpt ] || python -m transfer.ppo {job.dir}/ppo.json')
                ckpts = {n: f'{job.dir}/ppo/ppo_ep{n}.ckpt' for n in PPO_BUDGETS}
            else:
                method = 'bc' if m.startswith('filtered_bc') else 'awr'
                prev, data, ckpts = base, [], {}
                for r in range(max(ONLINE_BUDGETS) // ROUND_EPISODES):
                    rec = job.rollout(f'r{r}_rollouts', prev, 'left', ROUND_EPISODES,
                                      3_000_000 + 100_000 * s + ROUND_EPISODES * r, record=True)
                    data.append({'path': rec, 'success_only': method == 'bc'})
                    prev = job.train(f'r{r}', {'method': method, 'init_ckpt': prev, 'seed': s, 'steps': ROUND_STEPS,
                                               'lr': 1e-4, 'warmup': 100, 'batch_size': 1024, 'replay_frac': 0.5,
                                               'data': list(data) + replay_data})
                    if (r + 1) * ROUND_EPISODES in ONLINE_BUDGETS:
                        ckpts[(r + 1) * ROUND_EPISODES] = prev
            for n, ckpt in ckpts.items():
                job.rollout(f'n{n}_eval', ckpt, 'left,right', EVAL_N, EVAL_SEED)
                add_adapt_b(job, f'n{n}', ckpt, s)
            jobs.append(job.write())
    write_manifest(f"{stage}_k{base_k}_s{seeds.replace(',', '')}{tag}", jobs)


@cli.command()
@click.option('--base_k', required=True, type=int)
@click.option('--seed', default=0, type=int)
@click.option('--configs', required=True, help='JSON {name: PPO_SPEC overrides}')
@click.option('--episodes', default=1000, type=int)
def e2_dev(base_k, seed, configs, episodes):
    """PPO hyperparameter sweep on one seed: train, then evaluate on A and B."""
    jobs = []
    base = f'{RUNS}/e0/base_k{base_k}_s{seed}/base/final.ckpt'
    for name, overrides in json.loads(configs).items():
        job = Job(f'e2dev/k{base_k}_s{seed}/{name}')
        spec = {**PPO_SPEC, **overrides, 'init_ckpt': base, 'peg': 'left', 'seed': seed, 'episodes': episodes,
                'save_at': [episodes], 'env_meta_from': ENV_META, 'nut_range': json.loads(WIDE), 'out_dir': f'{job.dir}/ppo'}
        json.dump(spec, open(f'{job.dir}/ppo.json', 'w'), indent=1)
        job.lines.append(f'[ -f {job.dir}/ppo/ppo_ep{episodes}.ckpt ] || python -m transfer.ppo {job.dir}/ppo.json')
        job.rollout('eval', f'{job.dir}/ppo/ppo_ep{episodes}.ckpt', 'left,right', EVAL_N, EVAL_SEED)
        jobs.append(job.write())
    write_manifest(f'e2dev_k{base_k}_s{seed}', jobs)


# Robustness probes (rollout.py options); positions from the wide range unless the probe changes it
PROBES = {
    'push4': '--push_step 50 --push_dist 0.04',
    'push8': '--push_step 50 --push_dist 0.08',
    'friction0.1': '--friction_scale 0.1',
    'far': None,                                   # wider start range, below
}
FAR = '[[-0.35,-0.35],[0.15,0.35]]'


@cli.command()
@click.option('--base_k', required=True, type=int)
@click.option('--seeds', default='0,1,2')
@click.option('--arms', default='base,e1r/sft/n1000,e1r/filtered_bc/n1000,e1r/awr/n1000')
def e3(base_k, seeds, arms):
    """Robustness probes on A and B for finished checkpoints (one job per seed and arm)."""
    jobs = []
    for s in map(int, seeds.split(',')):
        for arm in arms.split(','):
            ckpt = (f'{RUNS}/e0/base_k{base_k}_s{s}/base/final.ckpt' if arm == 'base' else
                    f"{RUNS}/{arm.split('/')[0]}/k{base_k}_s{s}/{'/'.join(arm.split('/')[1:])}/final.ckpt")
            job = Job(f"e3/k{base_k}_s{s}/{arm.replace('/', '_')}")
            for name, opts in PROBES.items():
                out = f'{job.dir}/{name}.json'
                rng = FAR if name == 'far' else WIDE
                job.lines.append(f"[ -f {out} ] || python -m transfer.rollout --ckpt {ckpt} --pegs left,right --n {EVAL_N} "
                                 f"--seed_start {EVAL_SEED} --env_meta_from {ENV_META} --out_json {out} "
                                 f"--nut_range '{rng}'" + (f' {opts}' if opts else ''))
            jobs.append(job.write())
    write_manifest(f"e3_k{base_k}_s{seeds.replace(',', '')}", jobs)


# ======================================================================================
# Redesign (2026-10-12): mixed-quality data, original nut start range, data-matched A pool
#
# Base policies are pretrained (behavior cloning on every episode) on mixed-quality scripted
# data for both pegs: an imperfect operator that misjudges grasp and placement (gen_mixed.py,
# quality q in [0.3, 1], about a third succeed). Every offline method post-trains on the SAME
# pool of mixed-quality task-A episodes (q in [0, 1], about half succeed) and differs only in
# how it uses the reward. SFT on clean expert A demos is a reference with better data, and PPO
# is matched by episode count. Measured: task B zero-shot, and how well B is learned from 10
# clean B demos afterwards (fixed SFT recipe), plus A for context.
# ======================================================================================
MIXED_PRETRAIN = [f'{DATA}/pretrain_mixed_left.hdf5', f'{DATA}/pretrain_mixed_right.hdf5']
A_POOL = f'{DATA}/apool_left.hdf5'
EXPERT_A = f'{DATA}/sft_left.hdf5'          # clean scripted demos, original range
CLEAN_B = f'{DATA}/ceil_right.hdf5'
LEARN_B_K = 10                              # clean B demos for the learnability test
LEARN_B_STEPS = 2000
M_METHODS = ('sft_pool', 'filtered_bc', 'awr', 'udrl_scalar', 'udrl_axes', 'sft_expert')


def m_base_ckpt(s):
    return f'{RUNS}/m0/base_s{s}/base/final.ckpt'


def add_learn_b(job, tag, ckpt, seed, k=None, udrl_mode=None):
    """How well B is learned from here: SFT on k (default LEARN_B_K) clean B demos, then evaluate B.

    A UDRL policy is fine-tuned the UDRL way, on the demos' own return condition, and evaluated at
    the 90th percentile of it (all demos succeed), so its condition input stays meaningful."""
    k = k or LEARN_B_K
    name = f'{tag}_learnB' + ('' if k == LEARN_B_K else f'_k{k}')
    spec = {'method': 'bc', 'init_ckpt': ckpt, 'seed': seed, 'steps': LEARN_B_STEPS, 'lr': 1e-4, 'warmup': 100,
            'batch_size': 1024, 'data': [{'path': CLEAN_B, 'first_n': k}]}
    if udrl_mode:
        spec.update(method='udrl', udrl_mode=udrl_mode)
    learned = job.train(name, spec)
    job.rollout(f'{name}_eval', learned, 'right', EVAL_N, EVAL_SEED, nut_range=None)


def m_method_spec(m, n, common):
    if m == 'sft_pool':
        return {**common, 'method': 'bc', 'data': [{'path': A_POOL, 'first_n': n}]}
    if m == 'filtered_bc':
        return {**common, 'method': 'bc', 'data': [{'path': A_POOL, 'first_n': n, 'success_only': True}]}
    if m == 'awr':
        return {**common, 'method': 'awr', 'data': [{'path': A_POOL, 'first_n': n}]}
    if m.startswith('udrl'):
        return {**common, 'method': 'udrl', 'udrl_mode': m.split('_')[1], 'data': [{'path': A_POOL, 'first_n': n}]}
    assert m == 'sft_expert'
    return {**common, 'method': 'bc', 'data': [{'path': EXPERT_A, 'first_n': n}]}


@cli.command()
@click.option('--seeds', default='0,1,2')
def m0(seeds):
    """Base policies: behavior cloning on all mixed-quality pretraining episodes (both pegs)."""
    jobs = []
    for s in map(int, seeds.split(',')):
        job = Job(f'm0/base_s{s}')
        ckpt = job.train('base', {'method': 'bc', 'seed': s, 'steps': BASE_STEPS, 'lr': 2e-4, 'warmup': 500,
                                  'batch_size': 1024, 'data': [{'path': p} for p in MIXED_PRETRAIN]})
        job.rollout('eval', ckpt, 'left,right', EVAL_N, EVAL_SEED, nut_range=None)
        add_learn_b(job, 'base', ckpt, s)
        jobs.append(job.write())
    write_manifest(f"m0_s{seeds.replace(',', '')}", jobs)


@cli.command()
@click.option('--seeds', default='0,1,2')
@click.option('--methods', default=','.join(M_METHODS))
@click.option('--replay/--no-replay', default=True, help='Mix the pretraining pool into post-training (50% of steps)')
@click.option('--learn_ks', default=str(LEARN_B_K), help='Numbers of clean B demos for the learnability test')
def m1(seeds, methods, replay, learn_ks):
    """Offline post-training on task A from the shared pool; evaluate A, B, and learning B."""
    stage = 'm1r' if replay else 'm1'
    jobs = []
    for s in map(int, seeds.split(',')):
        replay_data = [{'path': p, 'replay': True} for p in MIXED_PRETRAIN] if replay else []
        for m in methods.split(','):
            job = Job(f'{stage}/s{s}/{m}')
            for n in BUDGETS:
                common = {'init_ckpt': m_base_ckpt(s), 'seed': s, 'steps': POST_STEPS, 'lr': 1e-4, 'warmup': 200,
                          'batch_size': 1024, 'replay_frac': 0.5}
                spec = m_method_spec(m, n, common)
                spec['data'] = spec['data'] + replay_data
                ckpt = job.train(f'n{n}', spec)
                job.rollout(f'n{n}_eval', ckpt, 'left,right', EVAL_N, EVAL_SEED, nut_range=None)
                for k in map(int, learn_ks.split(',')):
                    add_learn_b(job, f'n{n}', ckpt, s, k, udrl_mode(m))
            jobs.append(job.write())
    write_manifest(f"{stage}_s{seeds.replace(',', '')}", jobs)


def udrl_mode(method):
    return method.split('_')[1] if method.startswith('udrl') else None


@cli.command()
@click.option('--stages', default='m0,m1r,m2')
@click.option('--seeds', default='0,1,2')
@click.option('--ks', default='3,5', help='Numbers of clean B demos to add (UDRL arms of m1r also get LEARN_B_K, '
                                       'which the first m1r run skipped)')
def m3(stages, seeds, ks):
    """More B-learnability points for finished checkpoints, one job per (stage, seed, arm)."""
    ks = [int(k) for k in ks.split(',')]
    jobs = []
    for stage in stages.split(','):
        for s in map(int, seeds.split(',')):
            if stage == 'm0':
                units = {'base': [('base', m_base_ckpt(s), None)]}
            elif stage == 'm2':
                units = {'ppo': [(f'n{n}', f'{RUNS}/m2/s{s}/ppo/ppo/ppo_ep{n}.ckpt', None) for n in PPO_BUDGETS]}
            else:
                units = {m: [(f'n{n}', f'{RUNS}/{stage}/s{s}/{m}/n{n}/final.ckpt', udrl_mode(m)) for n in BUDGETS]
                         for m in M_METHODS}
            for unit, ckpts in units.items():
                job = Job(f'm3/{stage}/s{s}/{unit}')
                for tag, ckpt, mode in ckpts:
                    unit_ks = ks + ([LEARN_B_K] if mode and stage == 'm1r' and LEARN_B_K not in ks else [])
                    for k in unit_ks:
                        add_learn_b(job, tag, ckpt, s, k, mode)
                src = Job(f'm0/base_s{s}' if stage == 'm0' else f'{stage}/s{s}/{unit}')
                jobs.append({**job.write(), 'source': {'name': src.name, 'dir': src.dir}})
    write_manifest(f"m3_{stages.replace(',', '')}_s{seeds.replace(',', '')}", jobs)


@cli.command()
@click.option('--seeds', default='0,1,2')
@click.option('--ppo_spec', default='{}', help='JSON overrides of PPO_SPEC')
@click.option('--tag', default='')
def m2(seeds, ppo_spec, tag):
    """PPO on task A from each base policy, matched by episode count; evaluate A, B, and learning B."""
    jobs = []
    for s in map(int, seeds.split(',')):
        job = Job(f'm2/s{s}/ppo{tag}')
        spec = {**PPO_SPEC, **json.loads(ppo_spec), 'init_ckpt': m_base_ckpt(s), 'peg': 'left', 'seed': s,
                'episodes': max(PPO_BUDGETS), 'save_at': list(PPO_BUDGETS), 'env_meta_from': ENV_META,
                'nut_range': None, 'out_dir': f'{job.dir}/ppo'}
        json.dump(spec, open(f'{job.dir}/ppo.json', 'w'), indent=1)
        job.lines.append(f'[ -f {job.dir}/ppo/ppo_ep{max(PPO_BUDGETS)}.ckpt ] || python -m transfer.ppo {job.dir}/ppo.json')
        for n in PPO_BUDGETS:
            ckpt = f'{job.dir}/ppo/ppo_ep{n}.ckpt'
            job.rollout(f'n{n}_eval', ckpt, 'left,right', EVAL_N, EVAL_SEED, nut_range=None)
            add_learn_b(job, f'n{n}', ckpt, s)
        jobs.append(job.write())
    write_manifest(f"m2_s{seeds.replace(',', '')}{tag}", jobs)


@cli.command()
@click.option('--seeds', default='0,1,2')
@click.option('--methods', default='filtered_bc,awr')
@click.option('--learn_ks', default='10,3,5', help="'' to skip the learnability test")
@click.option('--lr', default=1e-4)
@click.option('--round_steps', default=ROUND_STEPS)
@click.option('--episodes', default=max(ONLINE_BUDGETS))
@click.option('--eval_at', default=','.join(map(str, ONLINE_BUDGETS)), help='Episode counts to evaluate at')
@click.option('--tag', default='')
def m4(seeds, methods, learn_ks, lr, round_steps, episodes, eval_at, tag):
    """Online iterated filtered BC / AWR on task A, no replay: PPO's data source and episode budgets.

    Each round collects ROUND_EPISODES with the current policy and fine-tunes it on everything
    collected so far (successes only for filtered BC); checkpoints at ONLINE_BUDGETS episodes are
    evaluated like the other arms."""
    jobs = []
    for s in map(int, seeds.split(',')):
        for m in methods.split(','):
            job = Job(f'm4/s{s}/{m}{tag}')
            method = 'bc' if m == 'filtered_bc' else 'awr'
            prev, data = m_base_ckpt(s), []
            for r in range(episodes // ROUND_EPISODES):
                rec = job.rollout(f'r{r}_rollouts', prev, 'left', ROUND_EPISODES,
                                  3_000_000 + 100_000 * s + ROUND_EPISODES * r, record=True, nut_range=None)
                data.append({'path': rec, 'success_only': method == 'bc'})
                prev = job.train(f'r{r}', {'method': method, 'init_ckpt': prev, 'seed': s, 'steps': round_steps,
                                           'lr': lr, 'warmup': min(100, round_steps // 10), 'batch_size': 1024,
                                           'data': list(data)})
                n = (r + 1) * ROUND_EPISODES
                if n in map(int, eval_at.split(',')):
                    job.rollout(f'n{n}_eval', prev, 'left,right', EVAL_N, EVAL_SEED, nut_range=None)
                    for k in (map(int, learn_ks.split(',')) if learn_ks else []):
                        add_learn_b(job, f'n{n}', prev, s, k)
            jobs.append(job.write())
    write_manifest(f"m4_s{seeds.replace(',', '')}{tag}", jobs)


# name -> (learning rate, steps, save_every, steps at which to evaluate)
M5_SCHEDULES = {'1e-04': (1e-4, 1000, 100, (100, 200, 500, 1000)),
                '1e-05': (1e-5, 5000, 500, (500, 1000, 2500, 5000)),
                '1e-06': (1e-6, 10000, 1000, (1000, 3000, 6000, 10000)),
                '1e-06early': (1e-6, 1000, 100, (100, 200, 400, 700, 1000))}   # small changes on A


@cli.command()
@click.option('--seeds', default='0,1,2')
@click.option('--methods', default='filtered_bc,sft_expert')
@click.option('--lrs', default='1e-04,1e-05,1e-06', help=f'Names in M5_SCHEDULES: {",".join(M5_SCHEDULES)}')
def m5(seeds, methods, lrs):
    """Update-size control: supervised post-training on A (N=1000, no replay) at smaller learning rates,
    evaluated along the way, to compare its path in (change on A, change on B) with PPO's (lr 1e-6,
    KL-limited updates)."""
    jobs = []
    for s in map(int, seeds.split(',')):
        for m in methods.split(','):
            for name in lrs.split(','):
                lr, steps, every, evals = M5_SCHEDULES[name]
                job = Job(f'm5/s{s}/{m}_lr{name}')
                common = {'init_ckpt': m_base_ckpt(s), 'seed': s, 'steps': steps, 'lr': lr, 'warmup': min(200, steps // 10),
                          'batch_size': 1024, 'save_every': every}
                final = job.train('train', m_method_spec(m, 1000, common))
                for st in evals:
                    ckpt = final if st == steps else final.replace('final.ckpt', f'step{st}.ckpt')
                    job.rollout(f'step{st}_eval', ckpt, 'left,right', EVAL_N, EVAL_SEED, nut_range=None)
                jobs.append(job.write())
    default = '1e-04,1e-05,1e-06'
    write_manifest(f"m5_s{seeds.replace(',', '')}" + ('' if lrs == default else '_' + lrs.replace(',', '_')), jobs)


B_PPO_BUDGETS = (200, 1000, 3000)


def m6_starts(s):
    """Checkpoints PPO on B starts from. Arms whose B success is 0 (supervised without replay) are left out:
    with a sparse reward, PPO has nothing to learn from."""
    return {'base': m_base_ckpt(s),
            'ppoA6000': f'{RUNS}/m2/s{s}/ppo/ppo/ppo_ep6000.ckpt',
            'filtered_bc_r1000': f'{RUNS}/m1r/s{s}/filtered_bc/n1000/final.ckpt',
            'sft_expert_r1000': f'{RUNS}/m1r/s{s}/sft_expert/n1000/final.ckpt',
            'awr_r1000': f'{RUNS}/m1r/s{s}/awr/n1000/final.ckpt'}


@cli.command()
@click.option('--seeds', default='0,1,2')
@click.option('--starts', default='base,ppoA6000,filtered_bc_r1000,sft_expert_r1000,awr_r1000')
@click.option('--extra_start', multiple=True, help='name=checkpoint path with {s} for the seed, e.g. a KL-BC run')
def m6(seeds, starts, extra_start):
    """PPO on B (right peg) from checkpoints post-trained on A: does RL on A speed up RL on B?"""
    jobs = []
    for s in map(int, seeds.split(',')):
        pool = m6_starts(s)
        pool.update({kv.split('=', 1)[0]: kv.split('=', 1)[1].format(s=s) for kv in extra_start})
        names = starts.split(',') + [kv.split('=', 1)[0] for kv in extra_start]
        for name in names:
            job = Job(f'm6/s{s}/{name}')
            spec = {**PPO_SPEC, 'init_ckpt': pool[name], 'peg': 'right', 'seed': s, 'episodes': max(B_PPO_BUDGETS),
                    'save_at': list(B_PPO_BUDGETS), 'env_meta_from': ENV_META, 'nut_range': None, 'out_dir': f'{job.dir}/ppo'}
            json.dump(spec, open(f'{job.dir}/ppo.json', 'w'), indent=1)
            job.lines.append(f'[ -f {job.dir}/ppo/ppo_ep{max(B_PPO_BUDGETS)}.ckpt ] || python -m transfer.ppo {job.dir}/ppo.json')
            for n in B_PPO_BUDGETS:
                job.rollout(f'n{n}_eval', f'{job.dir}/ppo/ppo_ep{n}.ckpt', 'left,right', EVAL_N, EVAL_SEED, nut_range=None)
            jobs.append(job.write())
    write_manifest(f"m6_s{seeds.replace(',', '')}" + (f"_{'_'.join(kv.split('=')[0] for kv in extra_start)}"
                                                       if extra_start else ''), jobs)


@cli.command()
@click.option('--seeds', default='0,1,2')
@click.option('--methods', default='filtered_bc,sft_expert')
@click.option('--budgets', default='3000,6000', help="PPO episode counts whose KL from the base is the target")
@click.option('--kl_json', default=f'{ROOT}/results/m_kl.json', help='transfer.m_kl output with the PPO rows')
@click.option('--lr', default=1e-5)
@click.option('--steps', default=5000)
def m7(seeds, methods, budgets, kl_json, lr, steps):
    """BC on A without replay, held to PPO's KL from the base (KL through PPO's denoising chain, on A
    states, with an adaptive penalty): does staying as close to the base as PPO explain PPO's transfer?"""
    kl = json.load(open(kl_json))
    jobs = []
    for s in map(int, seeds.split(',')):
        for n in map(int, budgets.split(',')):
            target = next(r['kl_A'] for r in kl[f'PPO | {n} ep.'] if r['seed'] == s)
            for m in methods.split(','):
                job = Job(f'm7/s{s}/{m}_klppo{n}')
                common = {'init_ckpt': m_base_ckpt(s), 'seed': s, 'steps': steps, 'lr': lr, 'warmup': 200,
                          'batch_size': 1024, 'save_every': 500,
                          'kl': {'target': round(target, 4), 'sub_batch': 128, 'states': [MIXED_PRETRAIN[0]]}}
                final = job.train('train', m_method_spec(m, 1000, common))
                for st in (500, 1000, 2500, steps):
                    ckpt = final if st == steps else final.replace('final.ckpt', f'step{st}.ckpt')
                    job.rollout(f'step{st}_eval', ckpt, 'left,right', EVAL_N, EVAL_SEED, nut_range=None)
                jobs.append(job.write())
    write_manifest(f"m7_s{seeds.replace(',', '')}", jobs)


# ======================================================================================
# Second task pair (2026-10-13): A = the square nut onto the left peg, as before; B = the ROUND nut,
# grasped by the far side of its ring, onto the round (right) peg (round_task.py) - a different object,
# grasp point, wrist angle and peg. Same recipe: the base imitates both tasks' mixed-quality demos; the
# A data and arms are the first pair's. Evaluated on 'left' (A) and 'round' (B).
# ======================================================================================
X_PRETRAIN = [f'{DATA}/pretrain_mixed_left.hdf5', f'{DATA}/pretrain_mixed_round.hdf5']
X_PEGS = 'left,round'


def x_base_ckpt(s):
    return f'{RUNS}/x0/base_s{s}/base/final.ckpt'


@cli.command()
@click.option('--seeds', default='0,1,2')
def x0(seeds):
    """Base policies for the second pair: behavior cloning on both tasks' mixed-quality episodes."""
    jobs = []
    for s in map(int, seeds.split(',')):
        job = Job(f'x0/base_s{s}')
        ckpt = job.train('base', {'method': 'bc', 'seed': s, 'steps': BASE_STEPS, 'lr': 2e-4, 'warmup': 500,
                                  'batch_size': 1024, 'data': [{'path': p} for p in X_PRETRAIN]})
        job.rollout('eval', ckpt, X_PEGS, EVAL_N, EVAL_SEED, nut_range=None)
        jobs.append(job.write())
    write_manifest(f"x0_s{seeds.replace(',', '')}", jobs)


@cli.command()
@click.option('--seeds', default='0,1,2')
@click.option('--methods', default='filtered_bc,sft_expert')
def x1(seeds, methods):
    """Offline post-training on A (N=1000, 50% replay of both tasks' pretraining data), second pair."""
    jobs = []
    for s in map(int, seeds.split(',')):
        for m in methods.split(','):
            job = Job(f'x1r/s{s}/{m}')
            common = {'init_ckpt': x_base_ckpt(s), 'seed': s, 'steps': POST_STEPS, 'lr': 1e-4, 'warmup': 200,
                      'batch_size': 1024, 'replay_frac': 0.5}
            spec = m_method_spec(m, 1000, common)
            spec['data'] = spec['data'] + [{'path': p, 'replay': True} for p in X_PRETRAIN]
            ckpt = job.train('n1000', spec)
            job.rollout('n1000_eval', ckpt, X_PEGS, EVAL_N, EVAL_SEED, nut_range=None)
            jobs.append(job.write())
    write_manifest(f"x1r_s{seeds.replace(',', '')}", jobs)


@cli.command()
@click.option('--seeds', default='0,1,2')
def x2(seeds):
    """PPO on A from the second pair's bases; evaluated on A and on the round-nut task."""
    jobs = []
    budgets = (1000, 3000, 6000)
    for s in map(int, seeds.split(',')):
        job = Job(f'x2/s{s}/ppo')
        spec = {**PPO_SPEC, 'init_ckpt': x_base_ckpt(s), 'peg': 'left', 'seed': s, 'episodes': max(budgets),
                'save_at': list(budgets), 'env_meta_from': ENV_META, 'nut_range': None, 'out_dir': f'{job.dir}/ppo'}
        json.dump(spec, open(f'{job.dir}/ppo.json', 'w'), indent=1)
        job.lines.append(f'[ -f {job.dir}/ppo/ppo_ep{max(budgets)}.ckpt ] || python -m transfer.ppo {job.dir}/ppo.json')
        for n in budgets:
            job.rollout(f'n{n}_eval', f'{job.dir}/ppo/ppo_ep{n}.ckpt', X_PEGS, EVAL_N, EVAL_SEED, nut_range=None)
        jobs.append(job.write())
    write_manifest(f"x2_s{seeds.replace(',', '')}", jobs)


@cli.command()
@click.option('--seeds', default='0,1,2')
@click.option('--methods', default='sft_expert,filtered_bc')
def x5(seeds, methods):
    """Supervised post-training on A without replay at PPO's learning rate (1e-6), evaluated along
    the way, second pair."""
    jobs = []
    lr, steps, every, evals = M5_SCHEDULES['1e-06early']
    for s in map(int, seeds.split(',')):
        for m in methods.split(','):
            job = Job(f'x5/s{s}/{m}_lr1e-06early')
            common = {'init_ckpt': x_base_ckpt(s), 'seed': s, 'steps': steps, 'lr': lr, 'warmup': min(200, steps // 10),
                      'batch_size': 1024, 'save_every': every}
            final = job.train('train', m_method_spec(m, 1000, common))
            for st in evals:
                ckpt = final if st == steps else final.replace('final.ckpt', f'step{st}.ckpt')
                job.rollout(f'step{st}_eval', ckpt, X_PEGS, EVAL_N, EVAL_SEED, nut_range=None)
            jobs.append(job.write())
    write_manifest(f"x5_s{seeds.replace(',', '')}", jobs)


# ======================================================================================
# Third round (2026-10-13)
#  a*: bases that never see B (BC on A only; then PPO, or more supervised learning, on A). B is then
#      learned from clean demos (SFT on k of them) or by PPO, to compare how fast each start learns it.
#  xlearn, xppob: how fast the round-nut task is learned from the second pair's checkpoints
#  mlog, xlog: PPO with its training episodes logged; filtered BC / AWR trained on exactly those episodes
#  mtime, xtime: PPO with a time cost (pays for speed), logged; offline speed learners on its episodes
#  probes, xpeg: evaluation only: perturbations on both pairs, and the round peg moved
# Evaluations are separate CPU-only jobs (kind 'cpu'): they are bound by simulation, not the GPU.
# ======================================================================================
CLEAN_ROUND = f'{DATA}/ceil_round.hdf5'      # clean scripted round-nut demos (ring grasp), successes only
CLEAN = {'right': CLEAN_B, 'round': CLEAN_ROUND}
A_ALL_PEGS = 'left,right,round'
TIME_COST = 2.0                               # a full-length (500-step) episode costs 2; success pays 1
PPO_GAMMA_STEP = PPO_SPEC['gamma'] ** (1 / 8)      # PPO discounts per 8-step decision
LR_PATH = (1e-6, 10000, 1000, (1000, 3000, 6000, 10000))   # lr, steps, save every, evaluated steps (m5's 1e-06)
A_KS = (5, 10, 25, 50, 100)                   # clean B demos, for starts that never saw B
LOG_BUDGETS = (1000, 3000, 6000)
PPO_CPUS = 32


def eval_job(name, ckpt, pegs, after=(), extra=''):
    """CPU-only evaluation of one checkpoint on the evaluation start states -> runs/<name>/eval.json."""
    job = Job(name, after=after, kind='cpu')
    out = f'{job.dir}/eval.json'
    job.lines.append(f'[ -f {out} ] || python -m transfer.rollout --ckpt {ckpt} --pegs {pegs} --n {EVAL_N} '
                     f'--seed_start {EVAL_SEED} --env_meta_from {ENV_META} --out_json {out}'
                     + (f' {extra}' if extra else ''))
    return job.write()


def ppo_stage(name, init_ckpt, peg, seed, budgets, eval_pegs, after=(), **overrides):
    """A PPO job (training only) and a CPU evaluation job per saved checkpoint, run after it.
    Returns (manifest entries, PPO job name, PPO output dir)."""
    job = Job(name, after=after, cpus=PPO_CPUS)
    out = f'{job.dir}/ppo'
    spec = {**PPO_SPEC, **overrides, 'init_ckpt': init_ckpt, 'peg': peg, 'seed': seed, 'episodes': max(budgets),
            'save_at': list(budgets), 'env_meta_from': ENV_META, 'nut_range': None, 'out_dir': out}
    json.dump(spec, open(f'{job.dir}/ppo.json', 'w'), indent=1)
    job.lines.append(f'[ -f {out}/ppo_ep{max(budgets)}.ckpt ] || python -m transfer.ppo {job.dir}/ppo.json')
    jobs = [job.write()]
    jobs += [eval_job(f'{name}/eval_n{n}', f'{out}/ppo_ep{n}.ckpt', eval_pegs, after=[job.name]) for n in budgets]
    return jobs, job.name, out


def train_eval(name, spec, eval_pegs, after=(), path=False):
    """One supervised run (8 CPUs), then CPU evaluations: of the final checkpoint, or along LR_PATH."""
    job = Job(name, after=after, cpus=8)
    if not path:
        final = job.train('train', spec)
        return [job.write(), eval_job(f'{name}/eval', final, eval_pegs, after=[job.name])]
    lr, steps, every, evals = LR_PATH
    final = job.train('train', {**spec, 'steps': steps, 'lr': lr, 'warmup': 200, 'batch_size': 1024, 'save_every': every})
    jobs = [job.write()]
    for st in evals:
        ckpt = final if st == steps else final.replace('final.ckpt', f'step{st}.ckpt')
        jobs.append(eval_job(f'{name}/eval_step{st}', ckpt, eval_pegs, after=[job.name]))
    return jobs


def learn_b_spec(ckpt, seed, task, k):
    """SFT on the first k clean demos of B (LEARN_B_STEPS steps, lr 1e-4), as add_learn_b."""
    return {'method': 'bc', 'init_ckpt': ckpt, 'seed': seed, 'steps': LEARN_B_STEPS, 'lr': 1e-4, 'warmup': 100,
            'batch_size': 1024, 'data': [{'path': CLEAN[task], 'first_n': k}]}


def a_base_ckpt(s):
    return f'{RUNS}/a0/base_s{s}/train/final.ckpt'


def a_starts(s):
    return {'base': a_base_ckpt(s), 'ppoA6000': f'{RUNS}/a1/s{s}/ppo/ppo/ppo_ep6000.ckpt',
            'sft_expert': f'{RUNS}/a1/s{s}/sft_expert/train/final.ckpt',
            'filtered_bc': f'{RUNS}/a1/s{s}/filtered_bc/train/final.ckpt'}


@cli.command()
@click.option('--seeds', default='0,1,2')
@click.option('--ks', default=','.join(map(str, A_KS)))
def a(seeds, ks):
    """B never in pretraining. a0: BC on task A's mixed-quality pretraining episodes only (m0's A half),
    evaluated on A and, zero-shot, on both B tasks (whose goal input it never saw). a1: post-training on A
    by PPO (episodes logged), SFT on 1000 clean A demos, or filtered BC on the 1000-episode A pool (the usual
    recipe). a2: SFT on k clean B demos from each of these, for B = the right peg and B = the round nut."""
    jobs = []
    for s in map(int, seeds.split(',')):
        base_jobs = train_eval(f'a0/base_s{s}', {'method': 'bc', 'seed': s, 'steps': BASE_STEPS, 'lr': 2e-4,
                                                 'warmup': 500, 'batch_size': 1024,
                                                 'data': [{'path': MIXED_PRETRAIN[0]}]}, A_ALL_PEGS)
        jobs += base_jobs
        after = {'base': base_jobs[0]['name']}
        ppo, after['ppoA6000'], _ = ppo_stage(f'a1/s{s}/ppo', a_base_ckpt(s), 'left', s, LOG_BUDGETS, A_ALL_PEGS,
                                              after=[after['base']], log_rollouts=True)
        jobs += ppo
        common = {'init_ckpt': a_base_ckpt(s), 'seed': s, 'steps': POST_STEPS, 'lr': 1e-4, 'warmup': 200,
                  'batch_size': 1024}
        for m in ('sft_expert', 'filtered_bc'):
            arm = train_eval(f'a1/s{s}/{m}', m_method_spec(m, 1000, common), A_ALL_PEGS, after=[after['base']])
            jobs += arm
            after[m] = arm[0]['name']
        for start, ckpt in a_starts(s).items():
            for task in ('right', 'round'):
                job = Job(f'a2/s{s}/{start}/{task}', after=[after[start]], cpus=8)
                ckpts = {k: job.train(f'k{k}', learn_b_spec(ckpt, s, task, k)) for k in map(int, ks.split(','))}
                jobs.append(job.write())
                jobs += [eval_job(f'a2/s{s}/{start}/{task}/eval_k{k}', c, task, after=[job.name]) for k, c in ckpts.items()]
    write_manifest(f"a_s{seeds.replace(',', '')}", jobs)


@cli.command()
@click.option('--seeds', default='0,1,2')
@click.option('--starts', default='base,ppoA6000,sft_expert')
@click.option('--k_right', default=10, help='clean demos of the right-peg task before PPO (from stage a2)')
@click.option('--k_round', default=10, help='clean demos of the round-nut task before PPO (from stage a2)')
def a3(seeds, starts, k_right, k_round):
    """B never in pretraining: PPO on B from each A-pretrained start, after SFT on k clean B demos (a sparse
    reward needs some successes to start from); evaluated on A and B at 200 / 1000 / 3000 episodes."""
    jobs = []
    for s in map(int, seeds.split(',')):
        for start in starts.split(','):
            for task, k in (('right', k_right), ('round', k_round)):
                warm = f'{RUNS}/a2/s{s}/{start}/{task}/k{k}/final.ckpt'
                jobs += ppo_stage(f'a3/s{s}/{start}/{task}_k{k}', warm, task, s, B_PPO_BUDGETS, f'left,{task}',
                                  after=[f'a2-s{s}-{start}-{task}'])[0]
    write_manifest(f"a3_s{seeds.replace(',', '')}_k{k_right}_{k_round}", jobs)


def x_starts(s):
    return {'base': x_base_ckpt(s), 'ppoA6000': f'{RUNS}/x2/s{s}/ppo/ppo/ppo_ep6000.ckpt',
            'sft_expert_r1000': f'{RUNS}/x1r/s{s}/sft_expert/n1000/final.ckpt',
            'filtered_bc_r1000': f'{RUNS}/x1r/s{s}/filtered_bc/n1000/final.ckpt',
            'sft_expert_lr1e6_400': f'{RUNS}/x5/s{s}/sft_expert_lr1e-06early/train/step400.ckpt'}


def m_starts(s):
    return {'base': m_base_ckpt(s), 'ppoA6000': f'{RUNS}/m2/s{s}/ppo/ppo/ppo_ep6000.ckpt',
            'sft_expert_r1000': f'{RUNS}/m1r/s{s}/sft_expert/n1000/final.ckpt',
            'filtered_bc_r1000': f'{RUNS}/m1r/s{s}/filtered_bc/n1000/final.ckpt',
            'sft_expert_lr1e6_700': f'{RUNS}/m5/s{s}/sft_expert_lr1e-06early/train/step700.ckpt'}


@cli.command()
@click.option('--seeds', default='0,1,2')
@click.option('--ks', default='3,5,10')
def xlearn(seeds, ks):
    """How fast the round-nut task is learned (SFT on k clean round-nut demos) from the second pair's
    checkpoints."""
    jobs = []
    for s in map(int, seeds.split(',')):
        for start, ckpt in x_starts(s).items():
            job = Job(f'xlearn/s{s}/{start}', cpus=8)
            ckpts = {k: job.train(f'k{k}', learn_b_spec(ckpt, s, 'round', k)) for k in map(int, ks.split(','))}
            jobs.append(job.write())
            jobs += [eval_job(f'xlearn/s{s}/{start}/eval_k{k}', c, 'round', after=[job.name]) for k, c in ckpts.items()]
    write_manifest(f"xlearn_s{seeds.replace(',', '')}", jobs)


@cli.command()
@click.option('--seeds', default='0,1,2')
@click.option('--starts', default='base,ppoA6000,sft_expert_r1000')
def xppob(seeds, starts):
    """PPO on the round-nut task from the second pair's checkpoints: does RL on A speed up RL on B?"""
    jobs = []
    for s in map(int, seeds.split(',')):
        pool = x_starts(s)
        for start in starts.split(','):
            jobs += ppo_stage(f'xppob/s{s}/{start}', pool[start], 'round', s, B_PPO_BUDGETS, X_PEGS)[0]
    write_manifest(f"xppob_s{seeds.replace(',', '')}", jobs)


@cli.command()
@click.option('--pair', type=click.Choice(['m', 'x']), required=True)
@click.option('--seeds', default='0,1,2')
@click.option('--time_cost', default=0.0, help=f'0: PPO\'s usual reward (stage <pair>log); e.g. {TIME_COST}: <pair>time')
def logged(pair, seeds, time_cost):
    """PPO on A with its training episodes logged, then offline learners trained on exactly those episodes
    along LR_PATH (lr 1e-6), from the same base. Usual reward: filtered BC (successes) and AWR (PPO's reward
    and discount). With a time cost: filtered BC on the fastest 30% of successes, and AWR with the time cost."""
    stage = f"{pair}{'time' if time_cost else 'log'}"
    base_fn, pegs = (m_base_ckpt, 'left,right') if pair == 'm' else (x_base_ckpt, X_PEGS)
    jobs = []
    for s in map(int, seeds.split(',')):
        overrides = {'log_rollouts': True, **({'time_cost': time_cost} if time_cost else {})}
        ppo, ppo_name, out = ppo_stage(f'{stage}/s{s}/ppo', base_fn(s), 'left', s, LOG_BUDGETS, pegs, **overrides)
        jobs += ppo
        parts, prev = [], 0
        for n in LOG_BUDGETS:
            parts.append({'path': f'{out}/rollouts_{prev}_{n}.hdf5'})
            prev = n
        if time_cost:
            arms = {'fast_bc': {'method': 'bc', 'data': [{**p, 'success_only': True, 'fastest_frac': 0.3} for p in parts]},
                    'awr_time': {'method': 'awr', 'awr': {'gamma': PPO_GAMMA_STEP, 'time_cost': time_cost}, 'data': parts}}
        else:
            arms = {'filtered_bc': {'method': 'bc', 'data': [{**p, 'success_only': True} for p in parts]},
                    'awr': {'method': 'awr', 'awr': {'gamma': PPO_GAMMA_STEP}, 'data': parts}}
        for arm, spec in arms.items():
            jobs += train_eval(f'{stage}/s{s}/{arm}', {**spec, 'init_ckpt': base_fn(s), 'seed': s}, pegs,
                               after=[ppo_name], path=True)
    write_manifest(f"{stage}_s{seeds.replace(',', '')}", jobs)


PROBES_V2 = {'push4_50': '--push_step 50 --push_dist 0.04',       # nut moved 4 cm early in the approach
             'push4_120': '--push_step 120 --push_dist 0.04',     # ... around grasp time
             'friction0.1': '--friction_scale 0.1'}               # slippery nut


@cli.command()
@click.option('--seeds', default='0,1,2')
def probes(seeds):
    """Robustness probes (evaluation only) on both tasks of both pairs, for the base and the post-trained arms."""
    jobs = []
    for pair, starts_fn, pegs in (('m', m_starts, 'left,right'), ('x', x_starts, X_PEGS)):
        for s in map(int, seeds.split(',')):
            for arm, ckpt in starts_fn(s).items():
                for probe, opts in PROBES_V2.items():
                    jobs.append(eval_job(f'{pair}probe/s{s}/{arm}/{probe}', ckpt, pegs, extra=opts))
    write_manifest(f"probes_s{seeds.replace(',', '')}", jobs)


# Round-nut release, median offset from the round peg (runs/xrec): base (+1.02, -0.26) cm, PPO on A (+1.34, -0.54)
PEG_SHIFT_DIR = (0.32, -0.28)
PEG_SHIFTS_CM = (-0.425, 0.425, 0.85, 1.275, 1.7)       # along PEG_SHIFT_DIR; 0.425 cm = the base-to-PPO shift


@cli.command()
@click.option('--seeds', default='0,1,2')
@click.option('--arms', default='base,ppoA6000,sft_expert_r1000')
def xpeg(seeds, arms):
    """The round peg moved along PPO's release shift (evaluation only, round-nut task): if PPO's loss on B is
    its shifted release, its success curve over peg positions is the base's shifted by 0.425 cm."""
    norm = math.hypot(*PEG_SHIFT_DIR)
    u = (PEG_SHIFT_DIR[0] / norm, PEG_SHIFT_DIR[1] / norm)
    jobs = []
    for s in map(int, seeds.split(',')):
        pool = x_starts(s)
        for arm in arms.split(','):
            for t in PEG_SHIFTS_CM:
                dx, dy = t / 100 * u[0], t / 100 * u[1]
                jobs.append(eval_job(f'xpeg/s{s}/{arm}/t{t:+.3f}', pool[arm], 'round',
                                     extra=f'--peg2_shift {dx:.5f},{dy:.5f}'))
    write_manifest(f"xpeg_s{seeds.replace(',', '')}", jobs)


@cli.command()
@click.option('--name', required=True, help='stage name: jobs go to runs/<name>/<label>_s<seed>')
@click.option('--ckpt', 'ckpts', multiple=True, required=True, help='label=checkpoint path with {s} for the seed')
@click.option('--seeds', default='0,1,2')
@click.option('--pegs', default=X_PEGS)
@click.option('--n', default=EVAL_N)
def record(name, ckpts, seeds, pegs, n):
    """Recorded evaluation rollouts (same start states as the evaluations) for trajectory analysis."""
    jobs = []
    for s in map(int, seeds.split(',')):
        for kv in ckpts:
            label, path = kv.split('=', 1)
            job = Job(f'{name}/{label}_s{s}')
            job.rollout('rollouts', path.format(s=s), pegs, n, EVAL_SEED, record=True, nut_range=None)
            jobs.append(job.write())
    write_manifest(f"{name}_s{seeds.replace(',', '')}", jobs)


if __name__ == '__main__':
    cli()
