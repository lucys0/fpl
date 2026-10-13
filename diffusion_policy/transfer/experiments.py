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
    def __init__(self, name, after=()):
        self.dir = f'{RUNS}/{name}'
        self.name = name.replace('/', '-')
        self.after = list(after)
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
        return {'name': self.name, 'dir': self.dir, 'after': self.after}


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


if __name__ == '__main__':
    cli()
