"""
Shared pieces for the RL-transfer experiments on the two-peg square task.

Task A is the left peg, task B the right peg. Every policy gets the target peg
as a one-hot goal appended to each observation (left = [1, 0], right = [0, 1]),
optionally followed by extra conditioning values (upside-down RL).

Episodes (scripted demos and recorded rollouts alike) live in robomimic-style
hdf5 files: data/demo_<i>/obs/<key>, data/demo_<i>/actions (7-D axis-angle
absolute OSC actions), with per-episode attrs (target_peg, and for rollouts
success / final_peg / first_success_step / seed).
"""

import copy
import json

import gym
import h5py
import numpy as np
import torch

from diffusion_policy.common.normalize_util import (
    array_to_stats, robomimic_abs_action_only_normalizer_from_stat)
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler
from diffusion_policy.dataset.robomimic_replay_lowdim_dataset import normalizer_from_stat
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.policy.flow_transformer_lowdim_policy import FlowTransformerLowdimPolicy

OBS_KEYS = ['object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']
OBS_KEY_DIMS = [14, 3, 4, 2]
LOW_DIM = sum(OBS_KEY_DIMS)          # 23
GOAL_DIM = 2
ACTION_DIM = 10                      # pos (3) + rotation_6d (6) + gripper (1)
HORIZON, N_OBS_STEPS, N_ACTION_STEPS = 16, 2, 8
MAX_STEPS = 500
PEGS = ('left', 'right')
TASKS = ('left', 'right', 'round')           # round: the round nut onto the right (round) peg, see round_task.py
TASK_PEG = {'left': 'left', 'right': 'right', 'round': 'right'}       # the peg each task's nut must land on

_ROT = RotationTransformer(from_rep='axis_angle', to_rep='rotation_6d')


def goal_vec(peg):
    """Goal input; 'right' and 'round' (the B task of each task pair) share B's slot."""
    return np.array([1.0, 0.0] if peg == 'left' else [0.0, 1.0], dtype=np.float32)


def to_policy_action(action7):
    """7-D env action (axis-angle) -> 10-D policy action (rotation_6d)."""
    a = np.asarray(action7, dtype=np.float32)
    return np.concatenate([a[..., :3], _ROT.forward(a[..., 3:6]), a[..., 6:]], axis=-1).astype(np.float32)


def to_env_action(action10):
    """10-D policy action -> 7-D env action."""
    a = np.asarray(action10)
    return np.concatenate([a[..., :3], _ROT.inverse(a[..., 3:9]), a[..., 9:]], axis=-1)


# ---------------------------------------------------------------- policy

def build_policy(extra_cond_dim=0):
    """The paper's flow-matching transformer (train_flow_transformer_lowdim_workspace.yaml)."""
    obs_dim = LOW_DIM + GOAL_DIM + extra_cond_dim
    model = TransformerForDiffusion(
        input_dim=ACTION_DIM, output_dim=ACTION_DIM, horizon=HORIZON, n_obs_steps=N_OBS_STEPS,
        cond_dim=obs_dim, n_layer=8, n_head=4, n_emb=256, p_drop_emb=0.0, p_drop_attn=0.3,
        causal_attn=True, time_as_cond=True, obs_as_cond=True, n_cond_layers=0)
    return FlowTransformerLowdimPolicy(
        model=model, horizon=HORIZON, obs_dim=obs_dim, action_dim=ACTION_DIM,
        n_action_steps=N_ACTION_STEPS, n_obs_steps=N_OBS_STEPS, num_inference_steps=10,
        obs_as_cond=True, pred_action_steps_only=False, beta_a=1.5, beta_b=1.0)


def save_checkpoint(path, policy, ema_policy, extra_cond_dim, info):
    torch.save({'policy': policy.state_dict(), 'ema': ema_policy.state_dict(),
                'extra_cond_dim': extra_cond_dim, 'info': info}, path)


def load_checkpoint(path, device='cpu'):
    """Returns (policy, ema_policy, payload)."""
    payload = torch.load(path, map_location='cpu')
    policy = build_policy(payload['extra_cond_dim'])
    policy.load_state_dict(payload['policy'])
    ema_policy = build_policy(payload['extra_cond_dim'])
    ema_policy.load_state_dict(payload['ema'])
    return policy.to(device), ema_policy.to(device), payload


def widen_cond(policy, extra_cond_dim):
    """Copy of `policy` that takes `extra_cond_dim` more observation inputs.

    The new input columns of the observation embedding start at zero, so the
    widened policy computes exactly the same function until it is trained. The
    observation normalizer is one global scale, which carries over unchanged.
    """
    if extra_cond_dim == 0:
        return copy.deepcopy(policy)
    old_extra = policy.obs_dim - LOW_DIM - GOAL_DIM
    new = build_policy(old_extra + extra_cond_dim)
    state = policy.state_dict()
    w = state['model.cond_obs_emb.weight']
    state['model.cond_obs_emb.weight'] = torch.cat(
        [w, torch.zeros(w.shape[0], extra_cond_dim, dtype=w.dtype)], dim=1)
    obs_norm = policy.normalizer['obs']
    scale = obs_norm.params_dict['scale'].detach().cpu().numpy()
    offset = obs_norm.params_dict['offset'].detach().cpu().numpy()
    stats = {k: v.detach().cpu().numpy() for k, v in obs_norm.params_dict['input_stats'].items()}
    pad = lambda x, v: np.concatenate([x, np.full(extra_cond_dim, v, dtype=x.dtype)])
    normalizer = LinearNormalizer()
    normalizer['action'] = policy.normalizer['action']
    normalizer['obs'] = SingleFieldLinearNormalizer.create_manual(
        scale=pad(scale, scale[0]), offset=pad(offset, 0.0),
        input_stats_dict={'min': pad(stats['min'], 0.0), 'max': pad(stats['max'], 1.0),
                          'mean': pad(stats['mean'], 0.5), 'std': pad(stats['std'], 0.5)})
    for k in list(state):
        if k.startswith('normalizer.'):
            del state[k]
    missing, unexpected = new.load_state_dict(state, strict=False)
    assert all(k.startswith('normalizer.') for k in missing) and not unexpected, (missing, unexpected)
    new.set_normalizer(normalizer)
    return new


# ---------------------------------------------------------------- episodes

def load_episodes(path, pegs=None, max_per_peg=None, success_only=False, rng_seed=None):
    """Episodes from an hdf5 file as dicts with obs (T, 23), action (T, 10), attrs."""
    episodes = []
    with h5py.File(path, 'r') as f:
        keys = sorted(f['data'].keys(), key=lambda k: int(k.split('_')[1]))
        for k in keys:
            d = f['data'][k]
            attrs = {a: (v.decode() if isinstance(v, bytes) else v) for a, v in d.attrs.items()}
            attrs.setdefault('success', True)        # scripted demos are kept only on success
            if pegs is not None and attrs['target_peg'] not in pegs:
                continue
            if success_only and not attrs['success']:
                continue
            obs = np.concatenate([d['obs'][key][:] for key in OBS_KEYS], axis=-1).astype(np.float32)
            episodes.append({'obs': obs, 'action': to_policy_action(d['actions'][:]), 'attrs': attrs})
    if max_per_peg is not None:
        rng = np.random.RandomState(rng_seed)
        chosen = []
        for peg in PEGS:
            idx = [i for i, e in enumerate(episodes) if e['attrs']['target_peg'] == peg]
            rng.shuffle(idx)
            chosen += idx[:max_per_peg]
        episodes = [episodes[i] for i in sorted(chosen)]
    return episodes


def env_meta_from(path):
    with h5py.File(path, 'r') as f:
        return json.loads(f['data'].attrs['env_args'])


def write_episodes(path, episodes, env_args):
    """Write recorded rollouts in the same hdf5 layout as the scripted demos."""
    with h5py.File(path, 'w') as f:
        data = f.create_group('data')
        data.attrs['env_args'] = json.dumps(env_args)
        for i, ep in enumerate(episodes):
            g = data.create_group(f'demo_{i}')
            obs = g.create_group('obs')
            idx = 0
            for key, dim in zip(OBS_KEYS, OBS_KEY_DIMS):
                obs.create_dataset(key, data=ep['obs'][:, idx:idx + dim])
                idx += dim
            g.create_dataset('actions', data=ep['action7'])
            for a, v in ep['attrs'].items():
                g.attrs[a] = v


# ---------------------------------------------------------------- dataset

class EpisodeDataset(torch.utils.data.Dataset):
    """Training windows (obs + goal + extra conditioning, action[, weight])."""

    def __init__(self, episodes, cond_fn=None, weights=None):
        """cond_fn(ep) -> extra per-episode conditioning (or None); weights: per-step arrays."""
        rb = ReplayBuffer.create_empty_numpy()
        for i, ep in enumerate(episodes):
            T = len(ep['action'])
            cond = [goal_vec(ep['attrs']['target_peg'])]
            if cond_fn is not None:
                cond.append(np.asarray(cond_fn(ep), dtype=np.float32))
            cond = np.broadcast_to(np.concatenate(cond), (T, sum(len(c) for c in cond)))
            item = {'obs': np.concatenate([ep['obs'][:T], cond], axis=-1).astype(np.float32),
                    'action': ep['action'].astype(np.float32)}
            if weights is not None:
                item['weight'] = np.asarray(weights[i], dtype=np.float32)[:T, None]
            rb.add_episode(item)
        self.replay_buffer = rb
        self.sampler = SequenceSampler(replay_buffer=rb, sequence_length=HORIZON,
                                       pad_before=N_OBS_STEPS - 1, pad_after=N_ACTION_STEPS - 1)

    def get_normalizer(self):
        normalizer = LinearNormalizer()
        normalizer['action'] = robomimic_abs_action_only_normalizer_from_stat(
            array_to_stats(self.replay_buffer['action']))
        normalizer['obs'] = normalizer_from_stat(array_to_stats(self.replay_buffer['obs']))
        return normalizer

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx):
        return {k: torch.from_numpy(v) for k, v in self.sampler.sample_sequence(idx).items()}


# ---------------------------------------------------------------- env

class RecordingWrapper(gym.Wrapper):
    """Records every step (obs before the action, the 7-D action) and which peg the nut is on."""

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.traj = {'obs': [obs], 'action7': [], 'peg': []}
        self.first_peg, self.first_peg_step = 0, -1
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        rs = self.unwrapped.env.env                # TwoPeg wrapper -> robomimic env -> robosuite env
        nut = rs.sim.data.body_xpos[rs.obj_body_id[getattr(self.unwrapped, 'nut_name', 'SquareNut')]]
        peg = 1 if rs.on_peg(nut, 0) else (2 if rs.on_peg(nut, 1) else 0)   # 1 = left, 2 = right
        self.traj['action7'].append(np.asarray(action, dtype=np.float32))
        self.traj['obs'].append(obs)
        self.traj['peg'].append(peg)
        if peg and not self.first_peg:
            self.first_peg, self.first_peg_step = peg, len(self.traj['peg']) - 1
        return obs, reward, done, info

    def get_trajectory(self):
        return self.traj

    def get_first_peg(self):
        """(peg the nut first landed on: 0 none / 1 left / 2 right, the step it happened)"""
        return self.first_peg, self.first_peg_step


class ProbeWrapper(gym.Wrapper):
    """Physical perturbations for robustness probes: the nut's friction scaled by
    `friction_scale`, at env step `push_step` the nut moved `push_dist` meters in a
    random horizontal direction (drawn from np.random, which the env seed sets), and the
    right (round) peg moved by `peg2_shift` = (dx, dy) meters. The policy observes neither peg,
    so a moved peg changes only where the nut has to go, not what the policy sees."""

    def __init__(self, env, push_step=None, push_dist=0.0, friction_scale=1.0, peg2_shift=None):
        super().__init__(env)
        self.push_step, self.push_dist, self.friction_scale = push_step, push_dist, friction_scale
        self.peg2_shift = None if peg2_shift is None else np.asarray(peg2_shift, dtype=np.float64)
        self.nut = getattr(env.unwrapped, 'nut_name', 'SquareNut')
        self._friction0, self._peg2_pos0 = None, None

    def _robosuite(self):
        return self.unwrapped.env.env

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.t = 0
        rs = self._robosuite()
        m = rs.sim.model
        if self.friction_scale != 1.0:
            geoms = [i for i in range(m.ngeom) if m.geom_bodyid[i] == rs.obj_body_id[self.nut]]
            if self._friction0 is None:
                self._friction0 = m.geom_friction[geoms].copy()
            m.geom_friction[geoms] = self._friction0 * self.friction_scale
        if self.peg2_shift is not None:
            # a hard reset reloads the model, so set the position from the original every time
            if self._peg2_pos0 is None:
                self._peg2_pos0 = m.body_pos[rs.peg2_body_id].copy()
            m.body_pos[rs.peg2_body_id, :2] = self._peg2_pos0[:2] + self.peg2_shift
            rs.sim.forward()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.t += 1
        if self.push_step is not None and self.t == self.push_step:
            sim = self._robosuite().sim
            start, _ = sim.model.get_joint_qpos_addr(f'{self.nut}_joint0')
            angle = np.random.uniform(0, 2 * np.pi)
            sim.data.qpos[start:start + 2] += self.push_dist * np.array([np.cos(angle), np.sin(angle)])
            sim.forward()
            obs = self.env.get_observation()
        return obs, reward, done, info


def make_env_fn(env_meta, max_steps=MAX_STEPS, nut_range=None, push_step=None, push_dist=0.0, friction_scale=1.0,
                nut='square', peg2_shift=None):
    """Two-peg env, recorded, with diffusion policy's multi-step execution; nut 'round' for the round-nut task."""
    def env_fn():
        from diffusion_policy.env.robomimic.robomimic_twopeg_lowdim_wrapper import RobomimicTwoPegLowdimWrapper
        from diffusion_policy.env_runner.robomimic_lowdim_runner import create_env
        from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
        if nut == 'round':
            from transfer.round_task import RoundNutTwoPegWrapper, round_env_meta
            assert nut_range is None, 'the start range override is square-nut only'
            twopeg = RoundNutTwoPegWrapper(env=create_env(env_meta=round_env_meta(env_meta), obs_keys=OBS_KEYS),
                                           obs_keys=OBS_KEYS, init_state=None)
        else:
            twopeg = RobomimicTwoPegLowdimWrapper(env=create_env(env_meta=env_meta, obs_keys=OBS_KEYS),
                                                  obs_keys=OBS_KEYS, init_state=None)
        if nut_range is not None:
            lo, hi = np.asarray(nut_range[0]), np.asarray(nut_range[1])
            base_make = twopeg._make_init_state

            def make_init_state(rng=None, base_make=base_make):
                state = base_make(rng)
                state[10:12] = (rng or np.random).uniform(lo, hi, size=(2,))
                return state
            twopeg._make_init_state = make_init_state
        env = twopeg
        if push_step is not None or friction_scale != 1.0 or peg2_shift is not None:
            env = ProbeWrapper(twopeg, push_step=push_step, push_dist=push_dist, friction_scale=friction_scale,
                               peg2_shift=peg2_shift)
        return MultiStepWrapper(RecordingWrapper(env), n_obs_steps=N_OBS_STEPS,
                                n_action_steps=N_ACTION_STEPS, max_episode_steps=max_steps)
    return env_fn
