from __future__ import annotations
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv
from omni.isaac.lab.envs.mdp.commands import UniformVelocityCommand
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.envs.mdp.commands import UniformVelocityCommandCfg


class UniformVelocityCommandGaitLogging(UniformVelocityCommand):
    def __init__(self, cfg: UniformVelocityCommandGaitLoggingCfg, env: ManagerBasedEnv|ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.sensor_cfg = self.cfg.sensor_cfg
        self.sensor_cfg.resolve(self._env.scene)
        self.contact_sensor: ContactSensor = self._env.scene.sensors[self.sensor_cfg.name]
        self.metrics["foot_air_time_variance"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["foot_step_frequency"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["pair_1_step_frequency"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["pair_2_step_frequency"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["step_air_time"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["pair_1_air_time"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["pair_2_air_time"] = torch.zeros(self.num_envs, device=self.device)

    def _update_metrics(self):
        self.metrics["error_vel_xy"] = torch.linalg.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1)
        self.metrics["error_vel_yaw"] = torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
        last_air_time = self.contact_sensor.data.last_air_time[:, self.sensor_cfg.body_ids]
        self.metrics["foot_air_time_variance"] = torch.var(last_air_time, dim=1)

        gait_reward_name = "gait"
        gait_func = self._env.reward_manager.get_term_cfg(gait_reward_name).func
        if hasattr(gait_func, "valid_last_air_time") and gait_func.valid_last_air_time is not None:
            valid_last_air_time = gait_func.valid_last_air_time
            valid_last_air_time_env = torch.all(valid_last_air_time > 1.0e-6, dim=1)
            masked_valid_last_air_time = valid_last_air_time[valid_last_air_time_env]
            average_last_air_time = torch.mean(masked_valid_last_air_time)
            average_frequency = (1.0 / average_last_air_time / 2.0) if average_last_air_time > 0 else 0.0
            self.metrics["foot_step_frequency"][:] = average_frequency
            pair_1_average_last_air_time = torch.mean(masked_valid_last_air_time[:, [0, 1]])
            pair_1_average_frequency = (1.0 / pair_1_average_last_air_time / 2.0) if pair_1_average_last_air_time > 0 else 0.0
            self.metrics["pair_1_step_frequency"][:] = pair_1_average_frequency
            pair_2_average_last_air_time = torch.mean(masked_valid_last_air_time[:, [2, 3]])
            pair_2_average_frequency = (1.0 / pair_2_average_last_air_time / 2.0) if pair_2_average_last_air_time > 0 else 0.0
            self.metrics["pair_2_step_frequency"][:] = pair_2_average_frequency

            self.metrics["step_air_time"][:] = average_last_air_time if average_last_air_time > 0 else 0.0
            self.metrics["pair_1_air_time"][:] = pair_1_average_last_air_time if pair_1_average_last_air_time > 0 else 0.0
            self.metrics["pair_2_air_time"][:] = pair_2_average_last_air_time if pair_2_average_last_air_time > 0 else 0.0



@configclass
class UniformVelocityCommandGaitLoggingCfg(UniformVelocityCommandCfg):
    class_type: type = UniformVelocityCommandGaitLogging
    sensor_cfg: SceneEntityCfg=SceneEntityCfg("robot_contact_senosr", body_names=".*foot")


class UniformVelocityCommandGaitLoggingMultiSampling(UniformVelocityCommandGaitLogging):
    def __init__(self, cfg: UniformVelocityCommandGaitLoggingMultiSamplingCfg, env: ManagerBasedEnv|ManagerBasedRLEnv):
        super().__init__(cfg, env)
        
        self.cfg.previous_ranges.lin_vel_x = tuple(self.cfg.ranges.lin_vel_x)
        self.cfg.previous_ranges.lin_vel_y = tuple(self.cfg.ranges.lin_vel_y)
        self.cfg.previous_ranges.ang_vel_z = tuple(self.cfg.ranges.ang_vel_z)
        self.lin_vel_x_equal_ranges = True
        self.lin_vel_y_equal_ranges = True
        self.ang_vel_z_equal_ranges = True
        self.first_time_set_ranges = True

        self.lin_vel_x_sampling_ranges = torch.tensor([(self.cfg.ranges.lin_vel_x[0], self.cfg.previous_ranges.lin_vel_x[0]),
                                                   (self.cfg.previous_ranges.lin_vel_x[0], self.cfg.previous_ranges.lin_vel_x[1]),
                                                   (self.cfg.previous_ranges.lin_vel_x[1], self.cfg.ranges.lin_vel_x[1])], device=self.device)
        self.lin_vel_y_sampling_ranges = torch.tensor([(self.cfg.ranges.lin_vel_y[0], self.cfg.previous_ranges.lin_vel_y[0]),
                                                   (self.cfg.previous_ranges.lin_vel_y[0], self.cfg.previous_ranges.lin_vel_y[1]),
                                                   (self.cfg.previous_ranges.lin_vel_y[1], self.cfg.ranges.lin_vel_y[1])], device=self.device)
        self.ang_vel_z_sampling_ranges = torch.tensor([(self.cfg.ranges.ang_vel_z[0], self.cfg.previous_ranges.ang_vel_z[0]),
                                                   (self.cfg.previous_ranges.ang_vel_z[0], self.cfg.previous_ranges.ang_vel_z[1]),
                                                   (self.cfg.previous_ranges.ang_vel_z[1], self.cfg.ranges.ang_vel_z[1])], device=self.device)
        self.sampling_probs = torch.tensor([self.cfg.new_command_probs, 1.0 - 2 * self.cfg.new_command_probs, self.cfg.new_command_probs], device=self.device)
        
        self.vel_command_b_buffer = torch.zeros_like(self.vel_command_b)
        self.initial_zero_command_steps = self.cfg.initial_zero_command_steps
        self.binary_maximal_command = self.cfg.binary_maximal_command
        if self.binary_maximal_command:
            sample_scales = [-1, 0, 1]
            sample_scales = [-1, 1]
            sample_scales_len = len(sample_scales)
            self.maximal_command_sampling = torch.zeros((sample_scales_len**3, 3), device=self.device)
            index = 0
            for i in sample_scales:
                for j in sample_scales:
                    for k in sample_scales:
                        self.maximal_command_sampling[index] = torch.tensor([i, j, k], device=self.device)
                        index += 1

        self.metrics["lin_vel_x"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["lin_vel_y"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["ang_vel_z"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["initial_zero_command_steps"] = torch.ones(self.num_envs, device=self.device) * self.initial_zero_command_steps
        self.metrics["rel_standing_envs"] = torch.ones(self.num_envs, device=self.device) * self.cfg.rel_standing_envs

    def set_ranges(self, lin_vel_x: tuple[float, float] | None, lin_vel_y: tuple[float, float] | None, ang_vel_z: tuple[float, float] | None):
        if lin_vel_x is not None:
            self.cfg.previous_ranges.lin_vel_x = tuple(self.cfg.ranges.lin_vel_x)
            self.cfg.ranges.lin_vel_x = tuple(lin_vel_x)
            self.lin_vel_x_equal_ranges = self.cfg.previous_ranges.lin_vel_x == self.cfg.ranges.lin_vel_x
            self.lin_vel_x_sampling_ranges[:] = torch.tensor([(self.cfg.ranges.lin_vel_x[0], self.cfg.previous_ranges.lin_vel_x[0]),
                                                    (self.cfg.previous_ranges.lin_vel_x[0], self.cfg.previous_ranges.lin_vel_x[1]),
                                                    (self.cfg.previous_ranges.lin_vel_x[1], self.cfg.ranges.lin_vel_x[1])], device=self.device)
        if lin_vel_y is not None:
            self.cfg.previous_ranges.lin_vel_y = tuple(self.cfg.ranges.lin_vel_y)
            self.cfg.ranges.lin_vel_y = tuple(lin_vel_y)
            self.lin_vel_y_equal_ranges = self.cfg.previous_ranges.lin_vel_y == self.cfg.ranges.lin_vel_y
            self.lin_vel_y_sampling_ranges[:] = torch.tensor([(self.cfg.ranges.lin_vel_y[0], self.cfg.previous_ranges.lin_vel_y[0]),
                                                    (self.cfg.previous_ranges.lin_vel_y[0], self.cfg.previous_ranges.lin_vel_y[1]),
                                                    (self.cfg.previous_ranges.lin_vel_y[1], self.cfg.ranges.lin_vel_y[1])], device=self.device)
        if ang_vel_z is not None:
            self.cfg.previous_ranges.ang_vel_z = tuple(self.cfg.ranges.ang_vel_z)
            self.cfg.ranges.ang_vel_z = tuple(ang_vel_z)
            self.ang_vel_z_equal_ranges = self.cfg.previous_ranges.ang_vel_z == self.cfg.ranges.ang_vel_z
            self.ang_vel_z_sampling_ranges[:] = torch.tensor([(self.cfg.ranges.ang_vel_z[0], self.cfg.previous_ranges.ang_vel_z[0]),
                                                    (self.cfg.previous_ranges.ang_vel_z[0], self.cfg.previous_ranges.ang_vel_z[1]),
                                                    (self.cfg.previous_ranges.ang_vel_z[1], self.cfg.ranges.ang_vel_z[1])], device=self.device)
        print("^ " * 60)
        print("lin_vel_x_equal_ranges: ", self.lin_vel_x_equal_ranges)
        print("lin_vel_y_equal_ranges: ", self.lin_vel_y_equal_ranges)
        print("ang_vel_z_equal_ranges: ", self.ang_vel_z_equal_ranges)

        if self.lin_vel_x_equal_ranges and self.lin_vel_y_equal_ranges and self.ang_vel_z_equal_ranges:
            self.initial_zero_command_steps = self.cfg.final_initial_zero_command_steps
            self.cfg.rel_standing_envs = self.cfg.final_rel_standing_envs
            print("~ " * 60)
            print("Set initial_zero_command_steps to ", self.initial_zero_command_steps)
            print("Set rel_standing_envs to ", self.cfg.rel_standing_envs)

        self.first_time_set_ranges = False

    def _update_metrics(self):
        super()._update_metrics()
        self.metrics["lin_vel_x"][:] = self.cfg.ranges.lin_vel_x[1]
        self.metrics["lin_vel_y"][:] = self.cfg.ranges.lin_vel_y[1]
        self.metrics["ang_vel_z"][:] = self.cfg.ranges.ang_vel_z[1]
        self.metrics["initial_zero_command_steps"][:] = self.initial_zero_command_steps
        self.metrics["rel_standing_envs"][:] = self.cfg.rel_standing_envs
        last_air_time = self.contact_sensor.data.last_air_time[:, self.sensor_cfg.body_ids]
        self.metrics["foot_air_time_variance"] = torch.var(last_air_time, dim=1)

    def _resample_command(self, env_ids: Sequence[int]):
        if self.binary_maximal_command:
            sampling_indexed = torch.randint(0, self.maximal_command_sampling.shape[0], (len(env_ids),), device=self.device)
            maximal_commands = torch.tensor([self.cfg.ranges.lin_vel_x[1], self.cfg.ranges.lin_vel_y[1], self.cfg.ranges.ang_vel_z[1]], device=self.device)
            self.vel_command_b[env_ids] = self.maximal_command_sampling[sampling_indexed] * maximal_commands
        else:
            if self.lin_vel_x_equal_ranges and self.lin_vel_y_equal_ranges and self.ang_vel_z_equal_ranges:
                super()._resample_command(env_ids)
            else:
                r = torch.empty(len(env_ids), device=self.device)
                if self.lin_vel_x_equal_ranges:
                    self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
                else:
                    bin_indices = torch.multinomial(self.sampling_probs, len(env_ids), replacement=True)
                    for i, (low, high) in enumerate(self.lin_vel_x_sampling_ranges):
                        in_env_ids = bin_indices == i
                        if in_env_ids.any():
                            selected_envs = env_ids[torch.nonzero(in_env_ids, as_tuple=True)[0]]
                            self.vel_command_b[selected_envs, 0] = torch.empty(len(selected_envs), device=self.device).uniform_(low, high)
                if self.lin_vel_y_equal_ranges:
                    self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
                else:
                    bin_indices = torch.multinomial(self.sampling_probs, len(env_ids), replacement=True)
                    for i, (low, high) in enumerate(self.lin_vel_y_sampling_ranges):
                        in_env_ids = bin_indices == i
                        if in_env_ids.any():
                            selected_envs = env_ids[torch.nonzero(in_env_ids, as_tuple=True)[0]]
                            self.vel_command_b[selected_envs, 1] = torch.empty(len(selected_envs), device=self.device).uniform_(low, high)
                if self.ang_vel_z_equal_ranges:
                    self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
                else:
                    bin_indices = torch.multinomial(self.sampling_probs, len(env_ids), replacement=True)
                    for i, (low, high) in enumerate(self.ang_vel_z_sampling_ranges):
                        in_env_ids = bin_indices == i
                        if in_env_ids.any():
                            selected_envs = env_ids[torch.nonzero(in_env_ids, as_tuple=True)[0]]
                            self.vel_command_b[selected_envs, 2] = torch.empty(len(selected_envs), device=self.device).uniform_(low, high)
                # update standing envs
                self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

        # update buffer no matter what
        self.vel_command_b_buffer[env_ids] = self.vel_command_b[env_ids].clone()
        self._set_zero_command_for_beginning_steps()  # this is necessary for the first reset of the script

    def _update_command(self):
        self._set_zero_command_for_beginning_steps()
        self._recover_command_for_beginning_steps()
        super()._update_command()

    def _set_zero_command_for_beginning_steps(self):
        set_zero_command_envs = self._env.episode_length_buf < self.initial_zero_command_steps 
        if set_zero_command_envs.any():
            set_zero_command_envs = set_zero_command_envs.nonzero(as_tuple=True)[0]
            self.vel_command_b[set_zero_command_envs] = self.vel_command_b_buffer[set_zero_command_envs] * 0.0

    def _recover_command_for_beginning_steps(self):
        recover_command_envs = self._env.episode_length_buf == self.initial_zero_command_steps
        if recover_command_envs.any():
            recover_command_envs = recover_command_envs.nonzero(as_tuple=True)[0]
            self.vel_command_b[recover_command_envs] = self.vel_command_b_buffer[recover_command_envs].clone()


@configclass
class UniformVelocityCommandGaitLoggingMultiSamplingCfg(UniformVelocityCommandGaitLoggingCfg):
    class_type: type = UniformVelocityCommandGaitLoggingMultiSampling

    @configclass
    class PreviousRanges:
        lin_vel_x: tuple[float, float] = (0.0, 0.0)
        lin_vel_y: tuple[float, float] = (0.0, 0.0)
        ang_vel_z: tuple[float, float] = (0.0, 0.0)

    previous_ranges: PreviousRanges = PreviousRanges()
    new_command_probs: float = 0.15
    final_rel_standing_envs: float = 0.0
    initial_zero_command_steps: int = 0
    final_initial_zero_command_steps: int = 0
    binary_maximal_command: bool = False

