from __future__ import annotations
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, CurriculumTermCfg
import numpy as np
import math
import locotouch.mdp as mdp


class ModifyVelCommandsRangeBasedonReward(ManagerTermBase):
    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.current_command: mdp.UniformVelocityCommandGaitLoggingMultiSampling = env.command_manager.get_term(cfg.params['command_name'])
        self.current_command_ranges = self.current_command.cfg.ranges
        self.command_maximum_ranges = cfg.params['command_maximum_ranges']
        curriculum_bins = cfg.params['curriculum_bins']
        self.lin_vel_x_expansion = (self.command_maximum_ranges[0] - self.current_command_ranges.lin_vel_x[1]) / curriculum_bins[0]
        self.lin_vel_y_expansion = (self.command_maximum_ranges[1] - self.current_command_ranges.lin_vel_y[1]) / curriculum_bins[1]
        self.ang_vel_z_expansion = (self.command_maximum_ranges[2] - self.current_command_ranges.ang_vel_z[1]) / curriculum_bins[2]
        self.reset_envs_episode_length = cfg.params['reset_envs_episode_length'] * env.max_episode_length_s
        self.reward_name_lin = cfg.params['reward_name_lin']
        self.reward_name_ang = cfg.params['reward_name_ang']
        lin_vel_reward_term_cfg = env.reward_manager.get_term_cfg(self.reward_name_lin)
        ang_vel_reward_term_cfg = env.reward_manager.get_term_cfg(self.reward_name_ang)
        self.reward_threshold_lin = math.exp(-cfg.params['error_threshold_lin'] / lin_vel_reward_term_cfg.params["sigma"]) * lin_vel_reward_term_cfg.weight * env.max_episode_length_s
        self.reward_threshold_ang = math.exp(-cfg.params['error_threshold_ang'] / ang_vel_reward_term_cfg.params["sigma"]) * ang_vel_reward_term_cfg.weight * env.max_episode_length_s
        self.repeat_times_lin = cfg.params['repeat_times_lin']
        self.repeat_times_ang = cfg.params['repeat_times_ang']
        self.lin_forward_bins = 0
        self.ang_forward_bins = 0
        self.max_distance_bins = cfg.params['max_distance_bins']

        self.env_num = env.num_envs
        self.env_reseted_lin = torch.zeros(self.env_num, device=env.device, dtype=torch.bool)
        self.episode_length_buf_lin = torch.zeros(self.env_num, device=env.device, dtype=torch.float)
        self.episode_reward_sum_lin = torch.zeros(self.env_num, device=env.device, dtype=torch.float)
        self.success_repeat_times_lin = 0

        self.env_reseted_ang = torch.zeros(self.env_num, device=env.device, dtype=torch.bool)
        self.episode_length_buf_ang = torch.zeros(self.env_num, device=env.device, dtype=torch.float)
        self.episode_reward_sum_ang = torch.zeros(self.env_num, device=env.device, dtype=torch.float)
        self.success_repeat_times_ang = 0

    def __call__(self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        command_name: str = "base_velocity",
        command_maximum_ranges: list[float] = [0.6, 0.3, math.pi / 4],
        curriculum_bins: list[int] = [20, 20, 20],
        reset_envs_episode_length: float = 0.95,
        reward_name_lin: str = "track_lin_vel_xy",
        reward_name_ang: str = "track_ang_vel_z",
        error_threshold_lin: float = 0.05,
        error_threshold_ang: float = 0.08,
        repeat_times_lin: int = 5,
        repeat_times_ang: int = 5,
        max_distance_bins: int = 3,
        ):
        if (self.current_command_ranges.lin_vel_x[1] != self.command_maximum_ranges[0] or not self.current_command.lin_vel_x_equal_ranges \
            or self.current_command_ranges.lin_vel_y[1] != self.command_maximum_ranges[1] or not self.current_command.lin_vel_y_equal_ranges) \
            and self.lin_forward_bins - self.ang_forward_bins <= self.max_distance_bins:
            self.env_reseted_lin[env_ids] = True
            self.episode_length_buf_lin[env_ids] = env.episode_length_buf[env_ids].float()
            self.episode_reward_sum_lin[env_ids] = env.reward_manager._episode_sums[self.reward_name_lin][env_ids]
            if torch.all(self.env_reseted_lin) and torch.mean(self.episode_length_buf_lin) > self.reset_envs_episode_length and torch.mean(self.episode_reward_sum_lin) > self.reward_threshold_lin:
                self.success_repeat_times_lin += 1
                if self.success_repeat_times_lin == self.repeat_times_lin:
                    lin_vel_x_new_lower_bound = np.clip(self.current_command_ranges.lin_vel_x[0] - self.lin_vel_x_expansion, -self.command_maximum_ranges[0], 0.)
                    lin_vel_y_new_lower_bound = np.clip(self.current_command_ranges.lin_vel_y[0] - self.lin_vel_y_expansion, -self.command_maximum_ranges[1], 0.)
                    lin_vel_x_new_range = (lin_vel_x_new_lower_bound, -lin_vel_x_new_lower_bound)
                    lin_vel_y_new_range = (lin_vel_y_new_lower_bound, -lin_vel_y_new_lower_bound)
                    # print("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                    # print(f"Vx: {lin_vel_x_new_range}, Vy: {lin_vel_y_new_range}")
                    # print(f"Reset_envs lin_vel_reward: {torch.mean(self.episode_reward_sum_lin) / env.max_episode_length_s}")
                    self.current_command.set_ranges(lin_vel_x=lin_vel_x_new_range, lin_vel_y=lin_vel_y_new_range, ang_vel_z=None)
                    self.success_repeat_times_lin = 0
                    self.lin_forward_bins += 1
                self.env_reseted_lin[:] = False
                self.episode_length_buf_lin[:] = 0
                self.episode_reward_sum_lin[:] = 0
        if (self.current_command_ranges.ang_vel_z[1] != self.command_maximum_ranges[2] or not self.current_command.ang_vel_z_equal_ranges) \
            and self.ang_forward_bins - self.lin_forward_bins <= self.max_distance_bins:
            self.env_reseted_ang[env_ids] = True
            self.episode_length_buf_ang[env_ids] = env.episode_length_buf[env_ids].float()
            self.episode_reward_sum_ang[env_ids] = env.reward_manager._episode_sums[self.reward_name_ang][env_ids]

            if torch.all(self.env_reseted_ang) and torch.mean(self.episode_length_buf_ang) > self.reset_envs_episode_length and torch.mean(self.episode_reward_sum_ang) > self.reward_threshold_ang:
                self.success_repeat_times_ang += 1
                if self.success_repeat_times_ang == self.repeat_times_ang:
                    ang_vel_z_new_lower_bound = np.clip(self.current_command_ranges.ang_vel_z[0] - self.ang_vel_z_expansion, -self.command_maximum_ranges[2], 0.)
                    ang_vel_z_new_range = (ang_vel_z_new_lower_bound, -ang_vel_z_new_lower_bound)
                    # print("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                    # print(f"Wz: {ang_vel_z_new_range}")
                    # print(f"Reset_envs ang_vel_reward: {torch.mean(self.episode_reward_sum_ang) / env.max_episode_length_s}")
                    self.current_command.set_ranges(lin_vel_x=None, lin_vel_y=None, ang_vel_z=ang_vel_z_new_range)
                    self.success_repeat_times_ang = 0
                    self.ang_forward_bins += 1
                self.env_reseted_ang[:] = False
                self.episode_length_buf_ang[:] = 0
                self.episode_reward_sum_ang[:] = 0

