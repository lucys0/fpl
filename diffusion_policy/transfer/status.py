"""
One-line status of a stage: how many jobs are done, and which have left the
Slurm queue without their DONE marker (failed or cancelled).

  python -m transfer.status /iris/u/lucyshi/fpl_transfer/runs/e0_manifest.json
"""

import json
import os
import subprocess
import sys


def main(manifest):
    jobs = json.load(open(manifest))
    queued = set(subprocess.run(['squeue', '-u', os.environ['USER'], '-h', '-o', '%j'],
                                capture_output=True, text=True, check=True).stdout.split())
    done = [j['name'] for j in jobs if os.path.exists(f"{j['dir']}/DONE")]
    failed = [j['name'] for j in jobs if j['name'] not in done and j['name'] not in queued]
    print(f"done {len(done)}/{len(jobs)}" + (f" FAILED: {' '.join(failed)}" if failed else ''))


if __name__ == '__main__':
    main(sys.argv[1])
