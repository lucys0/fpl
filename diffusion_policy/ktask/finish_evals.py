"""
Evaluate a PPO run's checkpoints that have no eval_ep<N>.json yet (each in a fresh process, up to 3 tries).
ktask.ppo execs into this at the end, so no simulator state of the training process is left around
(an evaluation started next to a finished PPO process once failed in PhysX with CUDA error 700).

  python -m ktask.finish_evals RUN_DIR
"""

import json
import os
import subprocess
import sys


def main(run_dir):
    spec = json.load(open(os.path.join(run_dir, 'ppo.json')))
    ev = spec.get('eval') or {}
    failed = []
    for n_ep in sorted(spec.get('save_at', [spec['episodes']])):
        ckpt, out = os.path.join(run_dir, f'ppo_ep{n_ep}.ckpt'), os.path.join(run_dir, f'eval_ep{n_ep}.json')
        if os.path.exists(out) or not os.path.exists(ckpt):
            continue
        for attempt in range(3):
            r = subprocess.run([sys.executable, '-m', 'ktask.evaluate', ckpt, '-o', out, '--tasks',
                                str(ev.get('tasks', 'all')), '--n', str(ev.get('n', 128))],
                               stdout=open(out + '.log', 'w'), stderr=subprocess.STDOUT)
            if r.returncode == 0 and os.path.exists(out):
                print(f'evaluated {ckpt}', flush=True)
                break
            print(f'evaluation of {ckpt} failed (attempt {attempt + 1})', flush=True)
        else:
            failed.append(ckpt)
    if failed:
        print(f'{len(failed)} evaluations failed: {failed}', flush=True)
        sys.exit(1)
    print('all evaluations done', flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
