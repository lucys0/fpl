"""
Slurm jobs for the K-task experiments (stdlib only, so it runs on the login node).

A job is a directory with steps.sh (its commands, then `touch DONE`); a manifest is a JSON list of
{name, dir, after}. `submit` tops up the queue: a job goes in once every job it depends on is DONE, it is
not DONE itself and not already queued (by name). iris-hi (higher priority tier) is filled first, up to
--max_hi, then iris up to --max_iris. A manifest's jobs run the code snapshot taken at its first submit
(<manifest>.snapshot), so later edits don't change a stage halfway.

  python -m ktask.jobs submit MANIFEST_OR_LIST.txt ... [--max_iris 18] [--max_hi 5] [--dry]
  python -m ktask.jobs status MANIFEST_OR_LIST.txt ...
"""

import argparse
import json
import os
import shutil
import subprocess
import time

ROOT = '/iris/u/lucyshi/fpl_transfer/ktask'
REPO = '/iris/u/lucyshi/fpl/diffusion_policy'
I = '/iris/u/lucyshi'
PY = f'{I}/maniskill/.venv/bin/python'
# iris4: small GPUs. iris[1-3] (TITAN RTX) and iris-hp-z8 (RTX 8000): PPO iterations 3-7 min vs ~1 min on
# L40S/H100 (2026-10-14), so a 64-iteration run can exceed the time limit
EXCLUDE = 'iris[1-4],iris-hp-z8'

WRAPPER = f'''#!/bin/bash
#SBATCH --account=iris
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --requeue
JOB_DIR=$1
export HOME={I} NETRC={I}/.netrc WANDB_CONFIG_DIR={I}/.config/wandb WANDB_CACHE_DIR={I}/.cache/wandb
export WANDB_DATA_DIR={I}/.local/share/wandb MPLCONFIGDIR={I}/.config/matplotlib WANDB__SERVICE_WAIT=300
export PYTHONPATH=SNAPSHOT PYTHONWARNINGS=ignore PATH={I}/maniskill/.venv/bin:$PATH
cd $JOB_DIR
# GPU check with a deadline: a hung driver blocks CUDA init forever instead of failing
python -c "import torch; (torch.ones(2, device='cuda') * 2).sum().item()" 2>/dev/null & CHECK=$!
for i in $(seq 60); do kill -0 $CHECK 2>/dev/null || break; sleep 5; done
if kill -0 $CHECK 2>/dev/null || ! wait $CHECK; then
    echo "$(date) no usable GPU on $(hostname -s)"
    hostname -s >> $JOB_DIR/bad_nodes
    exit 1
fi
echo "$(date) $(hostname -s) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
bash steps.sh
'''


class Job:
    def __init__(self, name, after=(), time='04:00:00'):
        self.dir = f'{ROOT}/runs/{name}'
        self.name = name.replace('/', '-')
        self.after = list(after)
        self.time = time
        self.lines = []

    def run(self, module, spec=None, args=''):
        """Add `python -m ktask.<module>`, with spec written to <dir>/<module>.json when given."""
        os.makedirs(self.dir, exist_ok=True)
        if spec is not None:
            path = f'{self.dir}/{module}.json'
            json.dump(spec, open(path, 'w'), indent=1)
            args = f'{path} {args}'.strip()
        self.lines.append(f'python -m ktask.{module} {args}')

    def parallel(self, runs):
        """Run several (module, spec_path, log_path) commands side by side on this job's GPU; fail if any fails."""
        for i, (module, spec_path, log_path) in enumerate(runs):
            self.lines.append(f'python -m ktask.{module} {spec_path} > {log_path} 2>&1 & P{i}=$!')
        self.lines.append(' && '.join(f'wait $P{i}' for i in range(len(runs))))

    def write(self):
        os.makedirs(self.dir, exist_ok=True)
        with open(f'{self.dir}/steps.sh', 'w') as f:
            f.write('set -e\n' + '\n'.join(self.lines) + f'\ntouch {self.dir}/DONE\n')
        return {'name': self.name, 'dir': self.dir, 'after': self.after, 'time': self.time}


def write_manifest(name, jobs):
    path = f'{ROOT}/runs/{name}_manifest.json'
    json.dump(jobs, open(path, 'w'), indent=1)
    print(f'{len(jobs)} jobs -> {path}')
    return path


def expand(args):
    out = []
    for a in args:
        out += [l.strip() for l in open(a) if l.strip() and not l.startswith('#')] if a.endswith('.txt') else [a]
    return out


def queued():
    """Names of my queued jobs -> partition."""
    out = subprocess.run(['squeue', '-u', os.environ.get('USER', 'lucyshi'), '-h', '-o', '%j %P'],
                         capture_output=True, text=True, check=True).stdout
    return dict(l.split() for l in out.splitlines() if l.strip())


def snapshot(manifest):
    snap_file = manifest.replace('_manifest.json', '.snapshot')
    if os.path.exists(snap_file):
        return open(snap_file).read().strip()
    tag = os.path.basename(manifest).replace('_manifest.json', '')
    dst = f"{ROOT}/code/{time.strftime('%Y%m%d-%H%M%S')}-{tag}"
    shutil.copytree(f'{REPO}/ktask', f'{dst}/ktask', ignore=shutil.ignore_patterns('__pycache__'))
    open(snap_file, 'w').write(dst)
    return dst


def done(d):
    return os.path.exists(f'{d}/DONE')


def submit(manifests, max_iris, max_hi, dry):
    q = queued()
    n_hi = sum(1 for p in q.values() if p == 'iris-hi')
    n_iris = sum(1 for p in q.values() if p == 'iris')
    jobs, dirs = [], {}
    for m in expand(manifests):
        for j in json.load(open(m)):
            dirs[j['name']] = j['dir']
            jobs.append((m, j))
    sent = 0
    for m, j in jobs:
        if j.get('external') or done(j['dir']) or j['name'] in q:     # external: runs elsewhere (iris-ws-4)
            continue
        if os.path.exists(f"{j['dir']}/CLAIMED"):                     # taken by a worker outside Slurm
            continue
        if any(not done(dirs.get(a, '/nonexistent')) for a in j['after']):
            continue
        if n_hi < max_hi:
            part, n_hi = 'iris-hi', n_hi + 1
        elif n_iris < max_iris:
            part, n_iris = 'iris', n_iris + 1
        else:
            break
        snap = snapshot(m)
        wrapper = f"{os.path.dirname(m)}/.wrapper_{os.path.basename(snap)}.sh"
        if not os.path.exists(wrapper):
            open(wrapper, 'w').write(WRAPPER.replace('SNAPSHOT', snap))
        cmd = ['sbatch', f'--partition={part}', f"--job-name={j['name']}", f"--time={j.get('time', '04:00:00')}",
               f'--exclude={EXCLUDE}', f"--output={j['dir']}/slurm-%j.out", wrapper, j['dir']]
        if dry:
            print(' '.join(cmd))
            continue
        for attempt in range(3):
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                break
            if 'QOSMaxSubmitJobPerUserLimit' in r.stderr:
                print(f'{part}: submit limit reached')
                return sent
            time.sleep(5)
        else:
            print(f"sbatch failed for {j['name']}: {r.stderr.strip()}")
            continue
        sent += 1
        print(f"{part}: {j['name']} ({r.stdout.strip()})")
    return sent


def status(manifests):
    q = queued()
    for m in expand(manifests):
        js = json.load(open(m))
        d = sum(done(j['dir']) for j in js)
        inq = sum(j['name'] in q for j in js)
        print(f'{os.path.basename(m)}: {d}/{len(js)} done, {inq} queued')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['submit', 'status'])
    ap.add_argument('manifests', nargs='+')
    ap.add_argument('--max_iris', type=int, default=18)
    ap.add_argument('--max_hi', type=int, default=5)
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    if a.cmd == 'submit':
        n = submit(a.manifests, a.max_iris, a.max_hi, a.dry)
        print(f'{time.strftime("%H:%M")} submitted {n}')
    else:
        status(a.manifests)
