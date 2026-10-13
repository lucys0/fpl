"""
Submit a stage's jobs (from its manifest) to Slurm, running a frozen copy of
the transfer code so later edits can't change jobs that are queued or running.

  python -m transfer.submit /iris/u/lucyshi/fpl_transfer/runs/e0_manifest.json --hi 6

The first --hi jobs in manifest order go to iris-hi (6-GPU cap), the rest to
iris (20-GPU cap). Jobs with a DONE marker are skipped; dependencies on them
are dropped. A job that finds no usable GPU records the node in
<job>/bad_nodes and holds itself; release it with --release_held.
"""

import json
import os
import shutil
import subprocess
import time

import click

ROOT = '/iris/u/lucyshi/fpl_transfer'
REPO = '/iris/u/lucyshi/fpl/diffusion_policy'
I = '/iris/u/lucyshi'
# iris4: small GPUs; iris8: GPU driver hung on 2026-10-12 (CUDA init blocks in the NVIDIA driver);
# iris-hgx-*: H100/H200 (sm_90), which the pinned torch 1.12 + CUDA 11.6 has no kernels for;
# iris1, iris9, iris10: one GPU each fails the GPU check (2026-10-12); being the free one, it got every
# new job. The rest are sc-freegpu nodes that don't mount /iris (orion-dgx) or weren't checked;
# macondo[1-2] and deep[17-25] were.
EXCLUDE = ('iris1,iris4,iris8,iris9,iris10,iris-hgx-1,iris-hgx-2,'
           'orion-dgx,dawn[1-8],dawn[10-18],britten[1-2],charlotte,macondo-dgx1,visionlab-dgx1')

WRAPPER = f'''#!/bin/bash
#SBATCH --account=iris
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
# Runs <job dir>/steps.sh with the code snapshot at SNAPSHOT.
JOB_DIR=$1
cd {REPO} && source .venv/bin/activate
export MUJOCO_PY_MUJOCO_PATH={I}/.mujoco/mujoco210 MUJOCO_PY_FORCE_CPU=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{I}/.mujoco/mujoco210/bin
export NETRC={I}/.netrc WANDB_CONFIG_DIR={I}/.config/wandb WANDB_CACHE_DIR={I}/.cache/wandb
export WANDB_DATA_DIR={I}/.local/share/wandb MPLCONFIGDIR={I}/.config/matplotlib WANDB__SERVICE_WAIT=300
export PYTHONPATH=SNAPSHOT:{REPO}
cd $JOB_DIR
# GPU check with a deadline: a hung driver blocks CUDA init forever instead of failing
python -c "import torch; (torch.ones(2, device='cuda') * 2).sum().item()" 2>/dev/null & CHECK=$!
for i in $(seq 60); do kill -0 $CHECK 2>/dev/null || break; sleep 5; done
if kill -0 $CHECK 2>/dev/null || ! wait $CHECK; then
    echo "$(date) no usable GPU on $(hostname -s); holding for a node exclusion"
    hostname -s >> $JOB_DIR/bad_nodes
    scontrol requeuehold $SLURM_JOB_ID
    sleep 120
    exit 1
fi
echo "$(date) $(hostname -s) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
bash steps.sh
'''


def snapshot_code():
    dst = f"{ROOT}/code/{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.copytree(f'{REPO}/transfer', f'{dst}/transfer', ignore=shutil.ignore_patterns('__pycache__'))
    return dst


@click.command()
@click.argument('manifest', required=False)
@click.option('--hi', default=0, type=int, help='How many jobs (in manifest order) go to iris-hi')
@click.option('--release_held', is_flag=True, help='Exclude recorded bad nodes from held jobs and release them')
@click.option('--after', default='', help='Comma-separated Slurm job ids every submitted job also waits for (afterok)')
@click.option('--cpus', default=None, type=int, help='Override CPUs per job (default 32), e.g. 64 for 60 PPO envs')
@click.option('--free', is_flag=True, help='Also allow the sc-freegpu partition (slower GPUs, 8-12 CPUs per node); '
                                           'implies 8 CPUs unless --cpus is given')
def main(manifest, hi, release_held, after, cpus, free):
    if release_held:
        out = subprocess.run(['squeue', '-u', os.environ['USER'], '-h', '-t', 'PD', '-o', '%i|%r'],
                             capture_output=True, text=True).stdout
        for line in out.splitlines():
            jobid, reason = line.split('|', 1)
            if 'held' not in reason.lower():  # JobHeldUser, or job_requeued_in_held_state after requeuehold
                continue
            info = subprocess.run(['scontrol', 'show', 'job', jobid], capture_output=True, text=True).stdout
            job_dir = os.path.dirname(info.split('StdOut=')[1].split()[0])   # logs go to <job dir>/slurm-%j.out
            bad = sorted(set(open(f'{job_dir}/bad_nodes').read().split())) if os.path.exists(f'{job_dir}/bad_nodes') else []
            subprocess.run(['scontrol', 'update', f'JobId={jobid}', f"ExcNodeList={','.join(EXCLUDE.split(',') + bad)}"], check=True)
            subprocess.run(['scontrol', 'release', jobid], check=True)
            print(f'released {jobid} ({job_dir}) excluding {bad}')
        return

    jobs = json.load(open(manifest))
    snap = snapshot_code()
    wrapper = f'{snap}/job.sbatch'
    open(wrapper, 'w').write(WRAPPER.replace('SNAPSHOT', snap))
    ids, submitted = {}, 0
    for job in jobs:
        if os.path.exists(f"{job['dir']}/DONE"):
            print(f"skip {job['name']} (done)")
            continue
        deps = [ids[a] for a in job['after'] if a in ids] + [a for a in after.split(',') if a]
        if 'source' in job and not os.path.exists(f"{job['source']['dir']}/DONE"):
            # a job from an earlier submission whose outputs this one needs: wait for it if it's still queued
            q = subprocess.run(['squeue', '-u', os.environ['USER'], '-h', '-n', job['source']['name'], '-o', '%i'],
                               capture_output=True, text=True).stdout.split()
            if not q:
                print(f"skip {job['name']}: {job['source']['name']} is neither done nor queued")
                continue
            deps += q
        partition = 'iris-hi' if submitted < hi else 'iris'
        if free:
            partition += ',sc-freegpu'
            cpus = cpus or 8
        cmd = ['sbatch', '--parsable', '-p', partition, f'--exclude={EXCLUDE}', '-J', job['name'],
               '-o', f"{job['dir']}/slurm-%j.out"]
        if deps:
            cmd.append('--dependency=afterok:' + ':'.join(deps))
        if cpus:
            cmd += [f'--cpus-per-task={cpus}', f'--mem={2 * cpus}G']
        r = subprocess.run(cmd + [wrapper, job['dir']], capture_output=True, text=True)
        if r.returncode and free and 'QOSMaxSubmitJobPerUserLimit' in r.stderr:   # iris submit cap: free partition only
            partition = 'sc-freegpu'
            cmd[cmd.index('-p') + 1] = partition
            r = subprocess.run(cmd + [wrapper, job['dir']], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"sbatch failed for {job['name']}: {r.stderr.strip()}")
        jobid = r.stdout.strip()
        ids[job['name']] = jobid
        submitted += 1
        print(f"{jobid} {partition:8s} {job['name']}" + (f" (after {','.join(deps)})" if deps else ''))
    json.dump(ids, open(manifest.replace('.json', f'_jobs_{time.strftime("%H%M%S")}.json'), 'w'), indent=1)
    print(f"{submitted} jobs submitted with code snapshot {snap}")


if __name__ == '__main__':
    main()
