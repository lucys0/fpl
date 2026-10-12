"""
Run scripts/collect_initial_scripted_rollouts.py with a different nut start
range. The script hardcodes x in [-0.2, 0], y in [-0.2, 0.2]; this swaps in a
reset that samples from --nut_range and passes every other argument through.

Usage:
  python -m transfer.gen_demos --nut_range "[[-0.3,-0.3],[0.1,0.3]]" -- -o OUT -n 25 --seed 1 --target_peg left ...
"""

import importlib.util
import json
import sys

import numpy as np

REPO = '/iris/u/lucyshi/fpl/diffusion_policy'


def main(argv):
    assert argv[0] == '--nut_range' and argv[2] == '--', __doc__
    lo, hi = (np.asarray(v, dtype=float) for v in json.loads(argv[1]))
    spec = importlib.util.spec_from_file_location('collector', f'{REPO}/scripts/collect_initial_scripted_rollouts.py')
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    from diffusion_policy.env.robomimic.robomimic_twopeg_lowdim_wrapper import _BASE_RESET_STATE

    def reset(self):
        # same as MultimodalSquareLowdimWrapper.reset, with the nut range swapped
        reset_state = _BASE_RESET_STATE.copy()
        reset_state[10:12] = np.random.uniform(lo, hi, size=(2,))
        self.rew = 0
        self.last_reset_state = reset_state.copy()
        self.env.env.reset_to({'states': reset_state})
        return self.get_observation()

    collector.MultimodalSquareLowdimWrapper.reset = reset
    collector.main(args=argv[3:], standalone_mode=True)


if __name__ == '__main__':
    main(sys.argv[1:])
