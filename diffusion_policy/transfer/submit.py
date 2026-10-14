"""
Submit a stage's jobs (from its manifest) to Slurm, running a frozen copy of
the transfer code so later edits can't change jobs that are queued or running.

  python -m transfer.submit /iris/u/lucyshi/fpl_transfer/runs/e0_manifest.json --hi 6

The first --hi jobs in manifest order go to iris-hi (6-GPU cap), the rest to
iris (20-GPU cap). Jobs with a DONE marker are skipped; dependencies on them
are dropped. A job that finds no usable GPU records the node in
<job>/bad_nodes and holds itself; release it with --release_held.

Manifest entries of kind 'cpu' (evaluations) run without a GPU on sc-freecpu,
which has no per-user job cap. With --top_up the command can be rerun at will
(e.g. every few minutes): it skips jobs that are done or already queued, waits
on queued dependencies by name, holds back jobs whose dependencies are not
submitted yet, and keeps at most --max_iris jobs in the iris QoS (submit cap
25) and --max_hi in iris-hi; the rest wait for the next round instead of
piling into one partition. A stage keeps the code snapshot of its first submit.
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
# sc-freecpu: napoli[1-16] and visionlab[17-25] mount /iris and run the env (checked 2026-10-13); orionp not checked
CPU_EXCLUDE = 'orionp[1-2]'
CPU_PARTITION = 'sc-freecpu'
# Pilots (transfer.pilot) on iris go to the nodes whose GPUs we can't use but whose CPUs sit idle, so they
# never take CPUs our GPU jobs need: iris8 (driver), iris9 and iris10 (a failing GPU), iris-hgx-* (sm_90)
PILOT_IRIS_EXCLUDE = 'iris[1-7],iris-hp-z8'


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


CPU_WRAPPER = f'''#!/bin/bash
#SBATCH --account=iris
#SBATCH --time=1-00:00:00
#SBATCH --requeue
# CPU only (evaluations): runs <job dir>/steps.sh with the code snapshot at SNAPSHOT.
JOB_DIR=$1
cd {REPO} && source .venv/bin/activate
export MUJOCO_PY_MUJOCO_PATH={I}/.mujoco/mujoco210 MUJOCO_PY_FORCE_CPU=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{I}/.mujoco/mujoco210/bin
export MPLCONFIGDIR={I}/.config/matplotlib PYTHONPATH=SNAPSHOT:{REPO}
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd $JOB_DIR
echo "$(date) $(hostname -s) CPU only, $(nproc) CPUs"
bash steps.sh
'''


PILOT_WRAPPER = f'''#!/bin/bash
#SBATCH --account=iris
#SBATCH --time=1-00:00:00
#SBATCH --requeue
# CPU only: a transfer.pilot that runs ready evaluations from the given manifests, with the code at SNAPSHOT.
cd {REPO} && source .venv/bin/activate
export MUJOCO_PY_MUJOCO_PATH={I}/.mujoco/mujoco210 MUJOCO_PY_FORCE_CPU=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{I}/.mujoco/mujoco210/bin
export MPLCONFIGDIR={I}/.config/matplotlib PYTHONPATH=SNAPSHOT:{REPO}
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd {ROOT}
echo "$(date) $(hostname -s) pilot, $(nproc) CPUs"
python -m transfer.pilot "$@"
'''


def ready_cpu_jobs(manifests):
    """Evaluations a pilot could start now: inputs done, not done, not failed twice, not freshly claimed."""
    n = 0
    for manifest in manifests:
        jobs = json.load(open(manifest))
        dirs = {j['name']: j['dir'] for j in jobs}
        for j in jobs:
            d = j['dir']
            if j.get('kind') != 'cpu' or os.path.exists(f'{d}/DONE'):
                continue
            if os.path.exists(f'{d}/.failed') and open(f'{d}/.failed').read().strip() not in ('', '0', '1'):
                continue
            if os.path.exists(f'{d}/.claim') and time.time() - os.path.getmtime(f'{d}/.claim') < 300:
                continue
            if all(a not in dirs or os.path.exists(f'{dirs[a]}/DONE') for a in j['after']):
                n += 1
    return n


def my_queue():
    """{job name: (job id, partition)} for my pending and running jobs."""
    out = subprocess.run(['squeue', '-u', os.environ['USER'], '-h', '-o', '%i|%j|%P'], capture_output=True, text=True).stdout
    q = {}
    for line in out.splitlines():
        jobid, name, part = line.split('|')
        q[name] = (jobid, part)
    return q


def stage_snapshot(manifest):
    """The code snapshot this stage's jobs run: made on its first submit, reused afterwards."""
    rec = manifest.replace('.json', '_snapshot.txt')
    if os.path.exists(rec):
        return open(rec).read().strip()
    snap = snapshot_code()
    open(f'{snap}/job.sbatch', 'w').write(WRAPPER.replace('SNAPSHOT', snap))
    open(f'{snap}/cpu.sbatch', 'w').write(CPU_WRAPPER.replace('SNAPSHOT', snap))
    open(rec, 'w').write(snap)
    return snap


def snapshot_code():
    dst = f"{ROOT}/code/{time.strftime('%Y%m%d-%H%M%S')}"
    n = 1
    while os.path.exists(dst):            # two snapshots within a second
        dst, n = f"{ROOT}/code/{time.strftime('%Y%m%d-%H%M%S')}-{n}", n + 1
    shutil.copytree(f'{REPO}/transfer', f'{dst}/transfer', ignore=shutil.ignore_patterns('__pycache__'))
    return dst


@click.command()
@click.argument('manifests', nargs=-1)
@click.option('--hi', default=0, type=int, help='How many jobs (in manifest order) go to iris-hi')
@click.option('--release_held', is_flag=True, help='Exclude recorded bad nodes from held jobs and release them')
@click.option('--after', default='', help='Comma-separated Slurm job ids every submitted job also waits for (afterok)')
@click.option('--cpus', default=None, type=int, help='Override CPUs per job (default 32), e.g. 64 for 60 PPO envs')
@click.option('--free', is_flag=True, help='Also allow the sc-freegpu partition (slower GPUs, 8-12 CPUs per node); '
                                           'implies 8 CPUs unless --cpus is given')
@click.option('--top_up', is_flag=True, help='Rerunnable submission within the caps below (see the module docstring)')
@click.option('--max_iris', default=24, help='--top_up: my jobs allowed in the iris QoS (its submit cap is 25)')
@click.option('--max_hi', default=8, help='--top_up: my jobs allowed in iris-hi (cap: 8 jobs, 6 GPUs); PPO goes there first')
@click.option('--pilots', default=0, help='--top_up: run CPU jobs through this many iris pilots (transfer.pilot) '
                                          'instead of one Slurm job each; they count against --max_iris')
@click.option('--pilot_cpus', default=64)
@click.option('--free_pilots', default=0, help='--top_up: also this many pilots in sc-freecpu')
@click.option('--free_pilot_cpus', default=4)
def main(manifests, hi, release_held, after, cpus, free, top_up, max_iris, max_hi, pilots, pilot_cpus, free_pilots,
         free_pilot_cpus):
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

    pilot_args = list(manifests)                   # pilots get the original arguments (a .txt list is reread)
    manifests = [l.strip() for a in manifests for l in (open(a) if a.endswith('.txt') else [a]) if l.strip()]
    all_dirs = {j['name']: j['dir'] for m in manifests for j in json.load(open(m))}   # dependencies across manifests
    queue = my_queue() if top_up else {}
    use_pilots = top_up and (pilots or free_pilots)
    if use_pilots:
        max_iris -= pilots                 # keep room in the iris QoS for the pilots
    in_iris = sum('iris' in p.split(',') and not n.startswith('pilot-') for n, (_, p) in queue.items())
    in_hi = sum(p == 'iris-hi' for _, p in queue.values())
    total = 0
    for manifest in manifests:
        jobs = json.load(open(manifest))
        dirs = all_dirs
        snap = stage_snapshot(manifest) if top_up else snapshot_code()
        wrapper, cpu_wrapper = f'{snap}/job.sbatch', f'{snap}/cpu.sbatch'
        if not top_up:
            open(wrapper, 'w').write(WRAPPER.replace('SNAPSHOT', snap))
            open(cpu_wrapper, 'w').write(CPU_WRAPPER.replace('SNAPSHOT', snap))
        ids, submitted, waiting = {}, 0, 0
        for job in jobs:
            if os.path.exists(f"{job['dir']}/DONE"):
                if not top_up:
                    print(f"skip {job['name']} (done)")
                continue
            if job['name'] in queue or (use_pilots and job.get('kind') == 'cpu'):
                continue
            deps, ready = [a for a in after.split(',') if a], True
            for a in job['after']:
                if a in ids:
                    deps.append(ids[a])
                elif a in queue:
                    deps.append(queue[a][0])
                elif top_up and a in dirs and not os.path.exists(f"{dirs[a]}/DONE"):
                    ready = False          # its dependency isn't submitted yet: next round
            if top_up and job.get('kind') != 'cpu' and deps:
                ready = False              # GPU jobs wait unsubmitted until their inputs exist: the iris
                                           # submit cap counts pending jobs, so it goes to runnable ones
            if not ready:
                waiting += 1
                continue
            if 'source' in job and not os.path.exists(f"{job['source']['dir']}/DONE"):
                # a job from an earlier submission whose outputs this one needs: wait for it if it's still queued
                q = subprocess.run(['squeue', '-u', os.environ['USER'], '-h', '-n', job['source']['name'], '-o', '%i'],
                                   capture_output=True, text=True).stdout.split()
                if not q:
                    print(f"skip {job['name']}: {job['source']['name']} is neither done nor queued")
                    continue
                deps += q
            job_cpus = job.get('cpus') or cpus
            if job.get('kind') == 'cpu':
                partition, script = CPU_PARTITION, cpu_wrapper
                job_cpus = job_cpus or 8
                cmd = ['sbatch', '--parsable', '-p', partition, f'--exclude={CPU_EXCLUDE}']
            else:
                script = wrapper
                if top_up:
                    if (job_cpus or 32) >= 32 and in_hi < max_hi:
                        partition = 'iris-hi'
                    elif in_iris < max_iris:
                        partition = 'iris' + (',sc-freegpu' if free else '')
                    else:
                        waiting += 1
                        continue
                else:
                    partition = 'iris-hi' if submitted < hi else 'iris'
                    if free:
                        partition += ',sc-freegpu'
                if free:
                    job_cpus = job_cpus or 8
                cmd = ['sbatch', '--parsable', '-p', partition, f'--exclude={EXCLUDE}']
            cmd += ['-J', job['name'], '-o', f"{job['dir']}/slurm-%j.out"]
            if top_up:     # short limits let the scheduler backfill (PPO ~1-1.5 h, training and evaluation < 1 h)
                cmd.append('--time=' + ('04:00:00' if (job_cpus or 32) >= 32 and job.get('kind') != 'cpu' else '02:00:00'))
            if deps:
                cmd.append('--dependency=afterok:' + ':'.join(deps))
            if job_cpus:
                cmd += [f'--cpus-per-task={job_cpus}', f'--mem={2 * job_cpus}G']
            for attempt in range(3):          # the controller occasionally refuses a burst of submissions
                r = subprocess.run(cmd + [script, job['dir']], capture_output=True, text=True)
                if not r.returncode or 'QOSMaxSubmitJobPerUserLimit' in r.stderr:
                    break
                time.sleep(5)
            if r.returncode and free and job.get('kind') != 'cpu' and 'QOSMaxSubmitJobPerUserLimit' in r.stderr:
                if top_up:              # the cap counted more than we saw: try again next round
                    waiting += 1
                    in_iris = max_iris
                    continue
                partition = 'sc-freegpu'    # iris submit cap: free partition only
                cmd[cmd.index('-p') + 1] = partition
                r = subprocess.run(cmd + [script, job['dir']], capture_output=True, text=True)
            if r.returncode:
                if not top_up:
                    raise RuntimeError(f"sbatch failed for {job['name']}: {r.stderr.strip()}")
                print(f"sbatch failed for {job['name']} (next round): {r.stderr.strip() or r.returncode}")
                waiting += 1
                continue
            jobid = r.stdout.strip()
            ids[job['name']] = jobid
            queue[job['name']] = (jobid, partition)
            in_hi += partition == 'iris-hi'
            in_iris += 'iris' in partition.split(',')
            submitted += 1
            print(f"{jobid} {partition:8s} {job['name']}" + (f" (after {','.join(deps)})" if deps else ''))
        if ids:
            json.dump(ids, open(manifest.replace('.json', f'_jobs_{time.strftime("%H%M%S")}.json'), 'w'), indent=1)
        total += submitted
        print(f"{os.path.basename(manifest)}: {submitted} submitted" + (f", {waiting} waiting" if top_up else '')
              + f" (code {snap})")
    if use_pilots:
        ready = ready_cpu_jobs(manifests)
        os.makedirs(f'{ROOT}/pilots', exist_ok=True)
        rec = f'{ROOT}/pilots/snapshot.txt'        # pilots run their own snapshot (it has transfer.pilot);
        if not os.path.exists(rec):                # evaluation code is the same in every stage's snapshot
            snap = snapshot_code()
            open(f'{snap}/pilot.sbatch', 'w').write(PILOT_WRAPPER.replace('SNAPSHOT', snap))
            open(rec, 'w').write(snap)
        script = f"{open(rec).read().strip()}/pilot.sbatch"
        want = [(f'pilot-iris-{k}', 'iris', PILOT_IRIS_EXCLUDE, pilot_cpus, 8) for k in range(pilots)]
        want += [(f'pilot-free-{k}', CPU_PARTITION, CPU_EXCLUDE, free_pilot_cpus, free_pilot_cpus) for k in range(free_pilots)]
        started = 0
        for name, part, exclude, n_cpus, slot in want:
            if name in queue or started >= ready:          # no more pilots than evaluations ready to run
                continue
            r = subprocess.run(['sbatch', '--parsable', '-p', part, f'--exclude={exclude}', '-J', name, '--time=03:00:00',
                                f'--cpus-per-task={n_cpus}', f'--mem={2 * n_cpus}G', '-o', f'{ROOT}/pilots/%x-%j.out',
                                script, *pilot_args, '--slot_cpus', str(slot), '--idle_exit', '2400'],
                               capture_output=True, text=True)
            if r.returncode:
                print(f"pilot {name} not submitted: {r.stderr.strip()}")
                continue
            started += n_cpus // slot
            print(f"{r.stdout.strip()} {part:8s} {name} ({n_cpus} CPUs)")
        print(f"evaluations ready: {ready}; pilots queued: {sum(n.startswith('pilot-') for n in queue)} before this round")
    if top_up:
        print(f"queue now: {in_iris} in iris, {in_hi} in iris-hi; {total} submitted this round")


if __name__ == '__main__':
    main()
