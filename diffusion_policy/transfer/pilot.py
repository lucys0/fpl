"""
Pilot job for the CPU-only evaluations: claims ready 'cpu' jobs from manifests (dependencies done, the job
neither done nor claimed by a live pilot) and runs them side by side, each pinned to its own share of this
job's CPUs, until none are left. One pilot replaces many one-evaluation Slurm jobs: the iris QoS caps
jobs, not CPUs, and the free CPU partition only has small gaps.

  python -m transfer.pilot runs/a_s012_manifest.json ... [--slot_cpus 8] [--idle_exit 1200]

A claim is <job dir>/.claim (pilot id), refreshed every minute while the job runs; a claim not refreshed
for 5 minutes (its pilot was preempted or killed) is taken over. A failed job is retried once, then left
with <job dir>/.failed. Jobs run in manifest order, so earlier manifests go first.
"""

import json
import os
import socket
import subprocess
import time

import click

STALE = 300          # seconds without a refresh before a claim counts as abandoned


def load(manifests):
    jobs, dirs = [], {}
    for m in manifests:
        for j in json.load(open(m)):
            dirs[j['name']] = j['dir']
            if j.get('kind') == 'cpu':
                jobs.append(j)
    return jobs, dirs


def failures(d):
    try:
        return int(open(f'{d}/.failed').read().strip() or 0)
    except (OSError, ValueError):
        return 0


def try_claim(d, me):
    path = f'{d}/.claim'
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            if time.time() - os.path.getmtime(path) < STALE:
                return False
            os.rename(path, f'{path}.stale.{me}')       # only one pilot wins the takeover
        except OSError:
            return False
        return try_claim(d, me)
    os.write(fd, me.encode())
    os.close(fd)
    return True


@click.command()
@click.argument('manifests', nargs=-1, required=True)
@click.option('--slot_cpus', default=8, help='CPUs per evaluation (it runs CPUs - 2 envs)')
@click.option('--idle_exit', default=1200, help='Exit after this many seconds with nothing to run')
def main(manifests, slot_cpus, idle_exit):
    me = f"{os.environ.get('SLURM_JOB_ID', 'local')}@{socket.gethostname().split('.')[0]}"
    cpus = sorted(os.sched_getaffinity(0))
    slots = [cpus[i:i + slot_cpus] for i in range(0, len(cpus) - slot_cpus + 1, slot_cpus)]
    print(f'{me}: {len(cpus)} CPUs, {len(slots)} slots of {slot_cpus}', flush=True)
    running = {}                                       # slot index -> (job dir, process)
    idle_since, last_beat = time.time(), 0.0
    while True:
        for k, (d, p) in list(running.items()):
            if p.poll() is None:
                continue
            if p.returncode:
                open(f'{d}/.failed', 'w').write(str(failures(d) + 1))
                print(f'{time.strftime("%H:%M")} FAILED ({p.returncode}) {d}', flush=True)
            else:
                print(f'{time.strftime("%H:%M")} done {d}', flush=True)
            try:
                os.remove(f'{d}/.claim')
            except OSError:
                pass
            del running[k]
        if time.time() - last_beat > 60:
            for d, _ in running.values():
                try:
                    os.utime(f'{d}/.claim')
                except OSError:
                    pass
            last_beat = time.time()
        jobs, dirs = load(manifests)                   # reread: manifests can be regenerated or added to
        pending = [j for j in jobs if not os.path.exists(f"{j['dir']}/DONE") and failures(j['dir']) < 2]
        free = [k for k in range(len(slots)) if k not in running]
        busy = {d for d, _ in running.values()}
        for j in pending:
            if not free:
                break
            if j['dir'] in busy or any(a in dirs and not os.path.exists(f'{dirs[a]}/DONE') for a in j['after']):
                continue
            if not try_claim(j['dir'], me):
                continue
            k = free.pop(0)
            log = open(f"{j['dir']}/pilot.log", 'a')
            log.write(f"\n# {time.strftime('%F %T')} {me} CPUs {slots[k][0]}-{slots[k][-1]}\n")
            log.flush()
            p = subprocess.Popen(['taskset', '-c', ','.join(map(str, slots[k])), 'bash', 'steps.sh'],
                                 cwd=j['dir'], stdout=log, stderr=subprocess.STDOUT)
            running[k] = (j['dir'], p)
            print(f"{time.strftime('%H:%M')} start {j['name']} on slot {k}", flush=True)
        if running:
            idle_since = time.time()
        elif not pending:
            print('all evaluations done', flush=True)
            return
        elif time.time() - idle_since > idle_exit:
            print(f'nothing runnable for {idle_exit} s; {len(pending)} still wait on their inputs', flush=True)
            return
        time.sleep(15)


if __name__ == '__main__':
    main()
