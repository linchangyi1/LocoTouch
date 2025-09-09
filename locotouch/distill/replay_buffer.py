import torch
from omni.isaac.lab.envs import ManagerBasedRLEnv
import numpy as np
from typing import Optional
from tqdm import tqdm
from .tactile_recorder import TactileRecorder


class ReplayBuffer:
    def __init__(self, env: ManagerBasedRLEnv, tactile_recorder: TactileRecorder, proprioception_dim: int):
        self._env = env
        self._num_envs = env.num_envs
        self._device = env.device
        self._proprioceptions, self._teacher_encoder_obses, self._tactile_signals, self._actions = [], [], [], []  # each data is a tensor for a full trajectory for one env, shape: (trajectory_length, obs_dim)
        self._proprioception_dim = proprioception_dim
        self._tactile_recorder = tactile_recorder
        self._steps_count = 0
        self._reward_sums = torch.zeros(self._num_envs, device=self._device)

    def collect_data(self, teacher_policy, student_policy: Optional[torch.nn.Module], num_steps: int):
        if student_policy is not None:  # when resetting the environment for two continuous times, the observations have some problems
            self._env.reset()
        self._tactile_recorder.reset()
        pbar = tqdm(total=num_steps, desc="Collecting Data", leave=True)
        trajectory_rewards = []
        trajectory_lengths = []
        proprioceptions, teacher_encoder_obses, tactile_signals, actions = [], [], [], []  # each data is a tensor for a single step for all envs, shape: (num_envs, obs_dim)
        steps_count = 0
        start_indices = torch.zeros(self._num_envs, device=self._device, dtype=torch.int64)
        start_count = self._steps_count

        with torch.no_grad():
        # with torch.inference_mode():
            proprioception_object_state, extras = self._env.get_observations()
            while self._steps_count - start_count < num_steps:
                # get observations and actions, then take a step
                proprioception = proprioception_object_state[:, :self._proprioception_dim]
                teacher_encoder_obs = proprioception_object_state[:, self._proprioception_dim:]
                tactile_signal = extras["observations"]["tactile"]
                action = teacher_policy(proprioception_object_state) if student_policy is None else student_policy(proprioception, tactile_signal)
                # store the data before the proprioception_object_state is updated!!
                proprioceptions.append(proprioception)
                teacher_encoder_obses.append(teacher_encoder_obs)
                self._tactile_recorder.record_new_tactile_signals(tactile_signal)
                tactile_signals.append(self._tactile_recorder.get_tactile_signals())
                # take a step
                proprioception_object_state, reward, dones, extras = self._env.step(action)
                self._reward_sums += reward.clone()
                steps_count += 1

                if dones.any():
                    if student_policy is not None:
                        student_policy.reset(dones)
                    done_idx = dones.nonzero(as_tuple=False).flatten()
                    self._tactile_recorder.reset(done_idx)
                    trajectory_rewards.extend(self._reward_sums[done_idx].cpu().tolist())
                    trajectory_lengths.extend((steps_count - start_indices[done_idx]).cpu().tolist())
                    self._reward_sums[done_idx] = 0
                    for env_id in done_idx:
                        if self._steps_count - start_count < num_steps:
                            self._record_new_traj(proprioceptions, teacher_encoder_obses, tactile_signals, actions, start_indices[env_id].item(), steps_count, env_id)
                            pbar.update(steps_count - start_indices[env_id].item())
                            start_indices[env_id] = steps_count
                        else:
                            break
        return trajectory_rewards, trajectory_lengths

    def _record_new_traj(self, proprioceptions, teacher_encoder_obses, tactile_signals, actions, start_idx, end_idx, env_id):
        if start_idx < end_idx:
            self._steps_count += (end_idx - start_idx)
            self._proprioceptions.append(torch.stack([p[env_id] for p in proprioceptions[start_idx:end_idx]]))
            self._teacher_encoder_obses.append(torch.stack([t[env_id] for t in teacher_encoder_obses[start_idx:end_idx]]))
            self._tactile_signals.append(torch.stack([t[env_id] for t in tactile_signals[start_idx:end_idx]]))
            # self._actions.append(torch.stack([a[env_id] for a in actions[start_idx:end_idx]]))

    def to_recurrent_generator(self, batch_size: int):
        num_trajs = len(self._proprioceptions)
        traj_indices = np.arange(num_trajs)
        traj_indices = np.random.permutation(traj_indices)
        for start_idx in range(0, num_trajs, batch_size):
            end_idx = np.minimum(start_idx + batch_size, num_trajs)
            yield self._prepare_padded_sequence(traj_indices[start_idx:end_idx])

    def _prepare_padded_sequence(self, traj_indices):
        traj_lengths = [self._proprioceptions[idx].shape[0] for idx in traj_indices]
        max_length = max(traj_lengths)
        num_trajs = len(traj_indices)

        # shape: (max_length, num_trajs, obs_dim)
        proprioceptions = torch.zeros((max_length, num_trajs, self._proprioceptions[0].shape[1]), device=self._device)
        teacher_encoder_obses = torch.zeros((max_length, num_trajs, self._teacher_encoder_obses[0].shape[1]), device=self._device)
        tactile_signals = torch.zeros((max_length, num_trajs, *self._tactile_signals[0].shape[1:]), device=self._device)
        # actions = torch.zeros((max_length, num_trajs, self._actions[0].shape[1]), device=self._device)
        masks = torch.zeros((max_length, num_trajs), dtype=torch.bool, device=self._device)
        for output_idx, traj_idx in enumerate(traj_indices):
            proprioceptions[:traj_lengths[output_idx], output_idx] = self._proprioceptions[traj_idx]
            tactile_signals[:traj_lengths[output_idx], output_idx] = self._tactile_signals[traj_idx]
            teacher_encoder_obses[:traj_lengths[output_idx], output_idx] = self._teacher_encoder_obses[traj_idx]
            # actions[:traj_lengths[output_idx], output_idx] = self._actions[traj_idx]
            masks[:traj_lengths[output_idx], output_idx] = 1

        return dict(proprioceptions=proprioceptions,
                    teacher_encoder_obses=teacher_encoder_obses,
                    tactile_signals=tactile_signals,
                    # actions=actions,
                    masks=masks)

    def clear_buffer(self):
        self._proprioceptions, self._teacher_encoder_obses, self._tactile_signals, self._actions = [], [], [], []
        self._steps_count = 0
        self._reward_sums[:] = 0

    def evaluate(self, student_policy, num_trajs: int):
        rewards = []
        lengths = []
        env_steps_count = torch.zeros(self._num_envs, device=self._device)
        with torch.no_grad():
            proprioception_object_state, extras = self._env.get_observations()
            while len(rewards) < num_trajs:
                proprioception = proprioception_object_state[:, :self._proprioception_dim]
                teacher_encoder_obs = proprioception_object_state[:, self._proprioception_dim:]
                tactile_signal = extras["observations"]["tactile"]
                action = student_policy(proprioception, tactile_signal)
                proprioception_object_state, reward, dones, extras = self._env.step(action)
                self._reward_sums += reward.clone()
                env_steps_count += 1
                if dones.any():
                    done_idx = dones.nonzero(as_tuple=False).flatten()
                    rewards.extend(self._reward_sums[done_idx].cpu().tolist())
                    lengths.extend(env_steps_count[done_idx].cpu().tolist())
                    self._reward_sums[done_idx] = 0
                    env_steps_count[done_idx] = 0
        return rewards, lengths

    @property
    def num_trajs(self):
        return len(self._proprioceptions)

    @property
    def num_steps(self):
        return self._steps_count

