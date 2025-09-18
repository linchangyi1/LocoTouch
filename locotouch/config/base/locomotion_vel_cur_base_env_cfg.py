import math
from isaaclab.utils import configclass
from isaaclab.managers import CurriculumTermCfg
import locotouch.mdp as mdp
from .locomotion_base_env_cfg import LocomotionBaseEnvCfg, smaller_scene_for_playing


@configclass
class LocomotionVelCurBaseEnvCfg(LocomotionBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # commands
        self.commands.base_velocity = mdp.UniformVelocityCommandGaitLoggingMultiSamplingCfg(
                asset_name=self.commands.base_velocity.asset_name,
                resampling_time_range=self.commands.base_velocity.resampling_time_range,
                rel_heading_envs=self.commands.base_velocity.rel_heading_envs,
                heading_command=self.commands.base_velocity.heading_command,
                debug_vis=self.commands.base_velocity.debug_vis,
                ranges=self.commands.base_velocity.ranges,
                # below parameters are specific for multi-sampling command
                new_command_probs=0.15,
                rel_standing_envs=0.1,
                final_rel_standing_envs = 0.1,
            )
        
        # curriculum
        self.curriculum.velocity_commands = CurriculumTermCfg(
            func=mdp.ModifyVelCommandsRangeBasedonReward,
            params={
                "command_name": "base_velocity",
                "command_maximum_ranges": [self.commands.base_velocity.ranges.lin_vel_x[1],
                                            self.commands.base_velocity.ranges.lin_vel_y[1],
                                            self.commands.base_velocity.ranges.ang_vel_z[1]],
                "curriculum_bins": [20, 20, 20],
                "reset_envs_episode_length": 0.98,
                "reward_name_lin": "track_lin_vel_xy",
                "reward_name_ang": "track_ang_vel_z",
                "error_threshold_lin": 0.056,
                "error_threshold_ang": 0.089,
                "repeat_times_lin": 1,
                "repeat_times_ang": 1,
                "max_distance_bins": 4,
            },
            )

        # initial commands
        self.commands.base_velocity.ranges.lin_vel_x = (-0.2, 0.2)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.1, 0.1)
        self.commands.base_velocity.ranges.ang_vel_z = (-math.pi / 10, math.pi / 10)


def locomotion_vel_cur_play_env_post_init_func(env_cfg: LocomotionVelCurBaseEnvCfg) -> None:
    smaller_scene_for_playing(env_cfg)
    velocity_commands_ranges = env_cfg.curriculum.velocity_commands.params["command_maximum_ranges"]
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (-velocity_commands_ranges[0], velocity_commands_ranges[0])
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (-velocity_commands_ranges[1], velocity_commands_ranges[1])
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (-velocity_commands_ranges[2], velocity_commands_ranges[2])
    env_cfg.commands.base_velocity.initial_zero_command_steps = env_cfg.commands.base_velocity.final_initial_zero_command_steps
    env_cfg.commands.base_velocity.rel_standing_envs = env_cfg.commands.base_velocity.final_rel_standing_envs

    # do it again for avoiding curriculum during play
    env_cfg.curriculum.velocity_commands.params["command_maximum_ranges"] = [env_cfg.commands.base_velocity.ranges.lin_vel_x[1],
                                                                            env_cfg.commands.base_velocity.ranges.lin_vel_y[1],
                                                                            env_cfg.commands.base_velocity.ranges.ang_vel_z[1]]


class LocomotionVelCurBaseEnvCfg_PLAY(LocomotionVelCurBaseEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        locomotion_vel_cur_play_env_post_init_func(self)

