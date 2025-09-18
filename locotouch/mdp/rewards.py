from __future__ import annotations
import math
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg, ManagerTermBase, RewardTermCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_from_euler_xyz, quat_apply, quat_apply_inverse, euler_xyz_from_quat, quat_inv, quat_mul
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
from locotouch.mdp.actions import JointPositionActionPrevPrev


# ----------------- Velocity Tracking Task -----------------
def track_lin_vel_xy_pst(
    env: ManagerBasedRLEnv, sigma: float=0.25, command_name: str="base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel_error = torch.linalg.norm((env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / sigma)

def track_ang_vel_z_pst(
    env: ManagerBasedRLEnv, sigma: float=0.25, command_name: str="base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel_error = torch.linalg.norm((env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2]).unsqueeze(1), dim=1)
    return torch.exp(-ang_vel_error / sigma)


# ----------------- Foot Slipping and Dragging -----------------
def foot_slipping_ngt(
    env: ManagerBasedRLEnv,
    threshold: float=1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*foot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("robot_contact_senosr", body_names=".*foot")
    ) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name] # type: ignore
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.linalg.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold # type: ignore
    foot_planar_velocity = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    return torch.sum(is_contact * foot_planar_velocity, dim=1)

def foot_dragging_ngt(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*foot"),
    height_threshold: float=0.025,
    foot_vel_xy_threshold: float=0.1,
    ) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_planar_velocity = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    is_moving = foot_planar_velocity > foot_vel_xy_threshold
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    near_ground = foot_height <= height_threshold
    is_dragging = torch.logical_and(near_ground, is_moving)
    return torch.sum(is_dragging, dim=1)


# ----------------- Gait -----------------
class AdaptiveSymmetricGaitReward(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.judge_time_threshold: float = cfg.params["judge_time_threshold"]
        self.air_time_gait_bound:float = cfg.params["air_time_gait_bound"]
        self.contact_time_gait_bound: float = cfg.params["contact_time_gait_bound"]
        self.async_time_tolerance: float = cfg.params["async_time_tolerance"]
        self.async_judge_time_threshold: float = self.judge_time_threshold + self.async_time_tolerance
        self.stance_rwd_scale: float = cfg.params["stance_rwd_scale"]
        self.encourage_symmetricity_and_low_frequency: bool = cfg.params["encourage_symmetricity_and_low_frequency"] > 0.5
        if self.encourage_symmetricity_and_low_frequency:
            soft_minimum_frequency: float = cfg.params["soft_minimum_frequency"]
            soft_max_air_time: float = 1.0 / (soft_minimum_frequency * 2.0)
            self.tolerance_proportion: float = cfg.params["tolerance_proportion"]
            self.rwd_upper_bound: float = cfg.params["rwd_upper_bound"]
            self.rwd_lower_bound: float = cfg.params["rwd_lower_bound"]
            self.vel_tracking_exp_sigma: float = cfg.params["vel_tracking_exp_sigma"]
            self.task_performance_ratio: float = cfg.params["task_performance_ratio"]
            self.linear_scale = self.rwd_upper_bound / soft_max_air_time  # the slope of the linear part
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]
        self.all_feet_ids = [self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1]]

        # for recording valid air time and contact time (don't record the time during zero command)
        feet_num = len(self.all_feet_ids)
        self.last_step_current_air_time = torch.zeros((env.scene.num_envs, feet_num), device=env.device)
        self.last_step_current_contact_time = torch.zeros((env.scene.num_envs, feet_num), device=env.device)
        self.swinging_in_zero_cmd = torch.zeros((env.scene.num_envs, feet_num), device=env.device, dtype=torch.bool)
        self.valid_last_air_time = torch.zeros((env.scene.num_envs, feet_num), device=env.device)
        self.valid_previous_contact = torch.zeros((env.scene.num_envs, feet_num), device=env.device, dtype=torch.bool)

        # record the velocity command
        self.last_velocity_cmd = torch.zeros((env.scene.num_envs, 3), device=env.device)
        self.vel_changed_judge = 1.0e-3
        self.step_from_changing_cmd = torch.zeros((env.scene.num_envs), device=env.device)

    def reset(self, env_ids = None):
        self.last_step_current_air_time[env_ids] = 0.0
        self.last_step_current_contact_time[env_ids] = 0.0
        self.swinging_in_zero_cmd[env_ids, :] = False
        self.valid_last_air_time[env_ids] = 0.0
        self.valid_previous_contact[env_ids] = False
        self.last_velocity_cmd[env_ids] = 0.0
        self.step_from_changing_cmd[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        synced_feet_pair_names,
        judge_time_threshold: float,
        air_time_gait_bound: float,
        contact_time_gait_bound: float,
        async_time_tolerance: float,
        stance_rwd_scale: float,
        encourage_symmetricity_and_low_frequency: float,
        soft_minimum_frequency: float,
        tolerance_proportion: float,
        rwd_upper_bound: float,
        rwd_lower_bound: float,
        vel_tracking_exp_sigma: float,
        task_performance_ratio: float,
    ) -> torch.Tensor:
        # update the valid last air time and contact time
        self._update_valid_last_air_contact_time(env)

        # reward for synchronous feet
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = (sync_reward_0 + sync_reward_1) / 2.0

        # reward for asynchronous feet
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = (async_reward_0 + async_reward_1 + async_reward_2 + async_reward_3) / 4.0

        # gait reward incudes stepping reward and stance reward
        stepping_reward = (sync_reward + async_reward) / 2.0
        stance_reward = self._stance_reward_func() * self.stance_rwd_scale
        non_zero_cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1) > 0.0
        gait_reward = torch.where(non_zero_cmd, stepping_reward, stance_reward)

        return gait_reward

    def _update_valid_last_air_contact_time(self, env: ManagerBasedRLEnv):
        # get air contact time
        current_air_time = self.contact_sensor.data.current_air_time[:, self.all_feet_ids]
        current_contact_time = self.contact_sensor.data.current_contact_time[:, self.all_feet_ids]
        last_air_time = self.contact_sensor.data.last_air_time[:, self.all_feet_ids]
        last_contact_time = self.contact_sensor.data.last_contact_time[:, self.all_feet_ids]
        
        # reset valid_last_air_time for zero_cmd_env
        non_zero_cmd_env = torch.norm(env.command_manager.get_command("base_velocity"), dim=1) > 0.0
        zero_cmd_env = torch.logical_not(non_zero_cmd_env)
        self.valid_last_air_time[zero_cmd_env, :] = 0.0

        # reset the new swinging feet
        new_swinging_feet = torch.logical_and(self.last_step_current_air_time<self.judge_time_threshold, current_air_time>self.judge_time_threshold)
        new_swinging_feet_with_non_zero_cmd = torch.logical_and(new_swinging_feet, non_zero_cmd_env.unsqueeze(-1))
        self.swinging_in_zero_cmd[new_swinging_feet_with_non_zero_cmd] = False
        
        # always check if the swing feet has experienced zero command
        swing_feet = current_air_time > self.judge_time_threshold
        swing_feet_expericing_zero_cmd = torch.logical_and(swing_feet, zero_cmd_env.unsqueeze(-1))
        self.swinging_in_zero_cmd[swing_feet_expericing_zero_cmd] = True

        # when changing command, make the same affet on valid_last_air_time as zero_cmd
        self.step_from_changing_cmd += 1
        current_vel_cmd = env.command_manager.get_command("base_velocity")
        changing_cmd = torch.any(torch.abs(current_vel_cmd - self.last_velocity_cmd) > self.vel_changed_judge, dim=1)
        self.last_velocity_cmd[changing_cmd] = current_vel_cmd[changing_cmd].clone()
        self.step_from_changing_cmd[changing_cmd] = 0
        self.swinging_in_zero_cmd[changing_cmd] = True  # swing feet expericing changing_cmd
        self.valid_last_air_time[changing_cmd, :] = 0.0

        # check new landing feet
        if torch.any(non_zero_cmd_env):
            new_landing_feet = torch.logical_and(self.last_step_current_contact_time<self.judge_time_threshold, current_contact_time>self.judge_time_threshold)
            valid_new_landing_feet = new_landing_feet & self.valid_previous_contact & (~self.swinging_in_zero_cmd)
            self.valid_last_air_time[valid_new_landing_feet] = last_air_time[valid_new_landing_feet].clone()

        # update the last air time and contact time
        self.last_step_current_air_time[:] = current_air_time.clone()
        self.last_step_current_contact_time[:] = current_contact_time.clone()

        # update the previous contact status, should be later then the aboved steps
        self.valid_previous_contact[current_contact_time>self.judge_time_threshold] = True

    def _vel_tracking_score(self) -> torch.Tensor:
        # bonus is conditioned on the tracking accuracy
        robot: RigidObject = self._env.scene["robot"]
        command_name = "base_velocity"
        vel_cmd = self._env.command_manager.get_command(command_name)
        non_zero_cmd = torch.norm(vel_cmd, dim=1) > 0.0
        lin_vel_error = torch.linalg.norm(vel_cmd[:, :2]-robot.data.root_lin_vel_b[:, :2], dim=1)
        ang_vel_error = torch.abs(vel_cmd[:, 2]-robot.data.root_ang_vel_b[:, 2])
        lin_vel_error_proportion = torch.where(non_zero_cmd, lin_vel_error, 0.0)
        ang_vel_error_proportion = torch.where(non_zero_cmd, ang_vel_error, 0.0)
        vel_tracking_score = (torch.exp(-lin_vel_error_proportion / self.vel_tracking_exp_sigma) + torch.exp(-ang_vel_error_proportion / self.vel_tracking_exp_sigma)) / 2.0
        return vel_tracking_score

    def _task_performance_score(self) -> torch.Tensor:
        return self._vel_tracking_score()

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        # sync air
        air_time = self.contact_sensor.data.current_air_time
        feet_air_time= air_time[:, [foot_0, foot_1]]
        both_feet_air = torch.all(
            (feet_air_time > self.judge_time_threshold) & (feet_air_time < self.air_time_gait_bound),
            dim=1)

        # sync contact
        contact_time = self.contact_sensor.data.current_contact_time
        foot_0_contact = torch.logical_and(contact_time[:, foot_0]>self.judge_time_threshold, contact_time[:, foot_0]<self.contact_time_gait_bound)
        foot_1_contact = torch.logical_and(contact_time[:, foot_1]>self.judge_time_threshold, contact_time[:, foot_1]<self.contact_time_gait_bound)
        both_feet_contact = torch.logical_and(foot_0_contact, foot_1_contact)

        if self.encourage_symmetricity_and_low_frequency:
            task_performance_score = self._task_performance_score()
            both_feet_air_rwd = self._swinging_bonus(foot_0, foot_1)
            rwd_scale = 1 - self.task_performance_ratio + self.task_performance_ratio * task_performance_score
            both_feet_air_rwd[both_feet_air_rwd>0.0] *= rwd_scale[both_feet_air_rwd>0.0]
            both_feet_air_rwd += 1.0
            return torch.where(both_feet_air, both_feet_air_rwd, torch.where(both_feet_contact, 1.0, 0.0))
        else:
            return torch.where(both_feet_air | both_feet_contact, 1.0, 0.0)

    def _swinging_bonus(self, foot_0: int, foot_1: int) -> torch.Tensor:
        air_time = self.contact_sensor.data.current_air_time
        feet_air_time= air_time[:, [foot_0, foot_1]]
        mean_feet_air_time = torch.mean(feet_air_time, dim=1)
        both_feet_air = torch.all(feet_air_time > self.judge_time_threshold, dim=1)

        # determine the feet with both valid air time pairs
        target_pair_feet_idex = [0, 1] if foot_0 in [self.all_feet_ids[0], self.all_feet_ids[1]] else [2, 3]
        valid_last_air_time_target_pair = self.valid_last_air_time[:, target_pair_feet_idex]
        mean_valid_last_air_time_target_pair = torch.mean(valid_last_air_time_target_pair, dim=1)
        valid_last_air_time_target_pair_env = torch.all(valid_last_air_time_target_pair > self.judge_time_threshold, dim=1)
        other_pair_feet_idex = [2, 3] if 0 in target_pair_feet_idex else [0, 1]
        valid_last_air_time_other_pair = self.valid_last_air_time[:, other_pair_feet_idex]
        mean_valid_last_air_time_other_pair = torch.mean(valid_last_air_time_other_pair, dim=1)
        valid_last_air_time_other_pair_env = torch.all(valid_last_air_time_other_pair > self.judge_time_threshold, dim=1)

        # add this to make sure the air time is larger than dt so that it's not a control-aware swinging
        valid_last_air_time_target_pair_env &= torch.all(valid_last_air_time_target_pair > 2*self._env.step_dt, dim=1)
        valid_last_air_time_other_pair_env &= torch.all(valid_last_air_time_other_pair > 2*self._env.step_dt, dim=1)

        # compute the reference air time
        either_pair_with_valid_last_air_time = valid_last_air_time_target_pair_env | valid_last_air_time_other_pair_env
        both_feet_air_with_reference_env = both_feet_air & either_pair_with_valid_last_air_time
        air_time_reference = torch.where(
            both_feet_air_with_reference_env,
            mean_valid_last_air_time_other_pair,
            0.0)

        # determine the extension and lower bound
        air_time_tolerance = air_time_reference + self.tolerance_proportion * air_time_reference
        last_air_time_difference = torch.where(
            both_feet_air_with_reference_env,
            mean_valid_last_air_time_target_pair - mean_valid_last_air_time_other_pair,
            0.0)
        air_time_extension = air_time_tolerance - last_air_time_difference
        air_time_extension = torch.clamp(air_time_extension, min=air_time_reference, max=air_time_tolerance)  # 0.0 <= air_time_extension <= air_time_tolerance

        # determine the air feet are within or beyond the extension
        both_feet_air_within_extension_env = both_feet_air_with_reference_env & (mean_feet_air_time <= air_time_extension)
        both_feet_air_between_extension_tolerance_env = both_feet_air_with_reference_env & (mean_feet_air_time > air_time_extension) & (mean_feet_air_time <= air_time_tolerance)

        # if difference is negative, always use the continuous line
        negative_diff_env = both_feet_air_with_reference_env & (last_air_time_difference < 0.0)
        both_feet_air_within_extension_env = both_feet_air_within_extension_env | negative_diff_env

        # compute the reward within reference
        both_feet_air_within_extension_rwd = torch.clamp(self.linear_scale * mean_feet_air_time, max=self.rwd_upper_bound)
        reference_point_rwd = torch.clamp(self.linear_scale * air_time_reference, max=self.rwd_upper_bound)
        extension_point_rwd = torch.clamp(self.linear_scale * air_time_extension, max=self.rwd_upper_bound)
        tolerance_point_rwd = torch.clamp(self.linear_scale * air_time_tolerance, max=self.rwd_upper_bound)

        # compute the reward between extension and tolerance
        extension_smaller_than_tolerance_env = both_feet_air_with_reference_env & (air_time_extension < air_time_tolerance)
        a_between = torch.where(
            extension_smaller_than_tolerance_env,
            -extension_point_rwd / (air_time_tolerance - air_time_extension),
            0.0)
        b_between = torch.where(
            extension_smaller_than_tolerance_env,
            - a_between * air_time_tolerance,
            0.0)
        both_feet_air_between_extension_tolerance_rwd = torch.where(
            extension_smaller_than_tolerance_env,
            a_between * mean_feet_air_time + b_between,
            extension_point_rwd)  # use extension_point_rwd when extension is equal to tolerance

        ## compute the reward beyond tolerance
        extension_larger_than_reference_env = both_feet_air_with_reference_env & (air_time_extension > air_time_reference)
        a_beyond = torch.where(
            extension_larger_than_reference_env,
            -reference_point_rwd / (air_time_extension - air_time_reference),
            0.0)
        b_beyond = torch.where(
            extension_larger_than_reference_env,
            - a_beyond * air_time_tolerance,
            0.0)
        swinging_rwd_lower_bound = torch.where(
            extension_smaller_than_tolerance_env,
            (last_air_time_difference / (self.tolerance_proportion * air_time_reference)) * self.rwd_lower_bound,
            tolerance_point_rwd)  # use tolerance_point_rwd when extension is equal to tolerance
        both_feet_air_zero_reference_env = both_feet_air_with_reference_env & ((~valid_last_air_time_other_pair_env))
        swinging_rwd_lower_bound[both_feet_air_zero_reference_env] = self.rwd_lower_bound  # add this for the case when the other pair has not valid last air time
        swinging_rwd_lower_bound = torch.clamp(swinging_rwd_lower_bound, min=self.rwd_lower_bound, max=self.rwd_upper_bound)
        both_feet_air_beyond_tolerance_rwd = torch.where(
            extension_larger_than_reference_env,
            a_beyond * mean_feet_air_time + b_beyond,
            swinging_rwd_lower_bound)  # use swinging_rwd_lower_bound when extension is equal to reference
        both_feet_air_beyond_tolerance_rwd = torch.clamp(both_feet_air_beyond_tolerance_rwd, min=swinging_rwd_lower_bound)

        both_feet_air_with_reference_rwd = torch.where(
            both_feet_air_within_extension_env,
            both_feet_air_within_extension_rwd,
            torch.where(
                both_feet_air_between_extension_tolerance_env,
                both_feet_air_between_extension_tolerance_rwd,
                both_feet_air_beyond_tolerance_rwd)
            )

        both_feet_air_rwd = torch.where(
            both_feet_air_with_reference_env,
            both_feet_air_with_reference_rwd,
            0.0)
        
        return both_feet_air_rwd

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # sync contact but within tolerance is ok
        foot_0_contact = torch.logical_and(contact_time[:, foot_0]>self.judge_time_threshold, contact_time[:, foot_0]<=self.async_judge_time_threshold)
        foot_1_contact = torch.logical_and(contact_time[:, foot_1]>self.judge_time_threshold, contact_time[:, foot_1]<=self.async_judge_time_threshold)
        both_contact = torch.logical_and(foot_0_contact, foot_1_contact)
        # async is ok
        foot_0_air = torch.logical_and(air_time[:, foot_0]>self.judge_time_threshold, air_time[:, foot_0]<self.air_time_gait_bound)
        foot_1_air = torch.logical_and(air_time[:, foot_1]>self.judge_time_threshold, air_time[:, foot_1]<self.air_time_gait_bound)
        foot_0_contact = torch.logical_and(contact_time[:, foot_0]>self.judge_time_threshold, contact_time[:, foot_0]<self.contact_time_gait_bound)
        foot_1_contact = torch.logical_and(contact_time[:, foot_1]>self.judge_time_threshold, contact_time[:, foot_1]<self.contact_time_gait_bound)
        air_contact = torch.logical_and(foot_0_air, foot_1_contact)
        contact_air = torch.logical_and(foot_0_contact, foot_1_air)

        return torch.where(both_contact | air_contact | contact_air, 1.0, 0.0)

    def _stance_reward_func(self) -> torch.Tensor:
        contact_time = self.contact_sensor.data.current_contact_time[:, self.all_feet_ids]
        all_stance = torch.all(contact_time > self.judge_time_threshold, dim=1)
        return torch.where(all_stance, 1.0, 0.0)


class AdaptiveSymmetricGaitRewardwithObject(AdaptiveSymmetricGaitReward):
    def _obj_balancing_score(self) -> torch.Tensor:
        robot: RigidObject = self._env.scene["robot"]
        obj: RigidObject = self._env.scene["object"]
        _, _, robot_yaw = euler_xyz_from_quat(robot.data.root_quat_w)
        robot_quat_only_yaw = quat_from_euler_xyz(torch.zeros_like(robot_yaw), torch.zeros_like(robot_yaw), robot_yaw)
        obj_xy_pos_world = obj.data.root_pos_w - robot.data.root_pos_w
        obj_xy_pos_robot_yaw = quat_apply_inverse(robot_quat_only_yaw, obj_xy_pos_world)
        obj_xy_pos = torch.abs(obj_xy_pos_robot_yaw[:, :2])
        x_max = self._env.reward_manager.get_term_cfg("object_dangerous_state").params["x_max"]
        y_max = self._env.reward_manager.get_term_cfg("object_dangerous_state").params["y_max"]
        obj_x_rwd_scale = torch.clip(1.0 - obj_xy_pos[:, 0] / x_max, min=0.0, max=1.0)
        obj_y_rwd_scale = torch.clip(1.0 - obj_xy_pos[:, 1] / y_max, min=0.0, max=1.0)
        obj_balancing_score = (obj_x_rwd_scale + obj_y_rwd_scale) / 2.0
        return obj_balancing_score
    
    def _task_performance_score(self) -> torch.Tensor:
        vel_tracking_score = self._vel_tracking_score()
        obj_balancing_score = self._obj_balancing_score()
        task_performance_score = (vel_tracking_score * 2 + obj_balancing_score) / 3.0
        task_performance_score = torch.clip(task_performance_score, min=0.0, max=1.0)
        return task_performance_score


# ----------------- Reguralization -----------------
# ------ Base (Torso) ------
def track_base_height_ngt(
    env: ManagerBasedRLEnv, target_height: float=0.26, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_pos_w[:, 2] - target_height)

def base_z_velocity_ngt(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])

def base_roll_pitch_velocity_ngt(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.root_ang_vel_b[:, :2]), dim=1)

def base_roll_pitch_angle_ngt(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)

# ------ Joint ------
def joint_position_limit_ngt(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    out_of_limits = -(asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]).clip(max=0.0)
    out_of_limits += (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)

def joint_position_ngt(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    stand_still_scale: float=5.0,
    velocity_threshold: float=0.3,
    ) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    final_reward = torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)
    return final_reward

def joint_velocity_ngt(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)

def joint_acceleration_ngt(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)

def joint_torque_ngt(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)

def action_rate_ngt(env: ManagerBasedRLEnv) -> torch.Tensor:
    action_term: JointPositionActionPrevPrev = env.action_manager.get_term("joint_pos") # type: ignore
    return torch.sum(torch.square(action_term.raw_actions - action_term.prev_raw_actions), dim=1)

# ------ Link ------
def thigh_calf_collision_ngt(
    env: ManagerBasedRLEnv, threshold: float=0.1,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("robot_contact_senosr", body_names=[".*thigh", ".*calf"])
    ) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name] # type: ignore
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.linalg.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold # type: ignore
    return torch.sum(is_contact, dim=1)

# ----------------- Object Transport -----------------
def object_relative_xy_position_ngt(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    work_only_when_cmd: int=0,
    ) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    plannar_distance = torch.linalg.norm((obj.data.root_pos_w - robot.data.root_pos_w)[:, :2], dim=1)  # world frame
    if bool(work_only_when_cmd):
        cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
        plannar_distance *= (cmd > 0.0)
    return plannar_distance

def object_relative_xy_velocity_ngt(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    lin_vel_in_robot_frame = quat_apply_inverse(robot.data.root_quat_w, obj.data.root_lin_vel_w - robot.data.root_lin_vel_w)
    return torch.sum(torch.square(lin_vel_in_robot_frame[:, :2]), dim=1)

def object_relative_z_velocity_ngt(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    lin_vel_in_robot_frame =  quat_apply_inverse(robot.data.root_quat_w, obj.data.root_lin_vel_w - robot.data.root_lin_vel_w)
    return torch.square(lin_vel_in_robot_frame[:, 2])

def object_relative_roll_pitch_angle_ngt(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    projected_gravity_w = quat_apply(obj.data.root_quat_w, obj.data.projected_gravity_b)
    projected_gravity_in_robot_frame = quat_apply_inverse(robot.data.root_quat_w, projected_gravity_w)
    return torch.sum(torch.square(projected_gravity_in_robot_frame[:, :2]), dim=1)

def object_relative_roll_pitch_velocity_ngt(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    ang_vel_in_robot_frame = quat_apply_inverse(robot.data.root_quat_w, obj.data.root_ang_vel_w - robot.data.root_ang_vel_w)
    return torch.sum(torch.abs(ang_vel_in_robot_frame[:, :2]), dim=1)

def object_relative_roll_angle_ngt(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    projected_gravity_w = quat_apply(obj.data.root_quat_w, obj.data.projected_gravity_b)
    projected_gravity_in_robot_frame = quat_apply_inverse(robot.data.root_quat_w, projected_gravity_w)
    return torch.square(projected_gravity_in_robot_frame[:, 1])

def object_relative_roll_velocity_ngt(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    ang_vel_in_robot_frame = quat_apply_inverse(robot.data.root_quat_w, obj.data.root_ang_vel_w - robot.data.root_ang_vel_w)
    return torch.square(ang_vel_in_robot_frame[:, 0])

def object_relative_yaw_angle_ngt(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    work_only_when_cmd: int=0,
    ) -> torch.Tensor:
    robot_asset: RigidObject | Articulation = env.scene[robot_cfg.name]
    object_asset: RigidObject | Articulation = env.scene[object_cfg.name]
    robot_quat = robot_asset.data.root_quat_w
    object_quat = object_asset.data.root_quat_w
    _, _, robot_yaw = euler_xyz_from_quat(robot_quat)
    _, _, object_yaw = euler_xyz_from_quat(object_quat)
    robot_quat_only_yaw = quat_from_euler_xyz(torch.zeros_like(robot_yaw), torch.zeros_like(robot_yaw), robot_yaw)
    object_quat_only_yaw = quat_from_euler_xyz(torch.zeros_like(object_yaw), torch.zeros_like(object_yaw), object_yaw)
    yaw_diff = euler_xyz_from_quat(quat_mul(quat_inv(robot_quat_only_yaw), object_quat_only_yaw))[2]
    yaw_diff[yaw_diff > torch.pi] -= 2 * torch.pi
    yaw_diff[yaw_diff > 0.5 * torch.pi] -= torch.pi  # (-pi, 0.5*pi]
    yaw_diff[yaw_diff <= -0.5 * torch.pi] += torch.pi  # (-0.5*pi, 0.5*pi]
    reward = torch.square(yaw_diff)
    if bool(work_only_when_cmd):
        cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
        reward *= (cmd > 0.0)
    return reward

def object_dangerous_state_ngt(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    x_max: float | None = None,
    y_max: float | None = None,
    z_min: float | None = None,
    roll_pitch_max: float | None = None,
    vel_xy_max: float | None = None,
    ) -> torch.Tensor:
    robot: RigidObject | Articulation = env.scene[robot_cfg.name]
    object: RigidObject | Articulation = env.scene[object_cfg.name]
    object_posistion_in_robot_frame = quat_apply_inverse(robot.data.root_quat_w, object.data.root_pos_w - robot.data.root_pos_w)
    object_in_danger = torch.zeros_like(object_posistion_in_robot_frame[:, 0], dtype=torch.bool)
    if x_max is not None:
        object_in_danger |= torch.abs(object_posistion_in_robot_frame[:, 0]) > x_max
    if y_max is not None:
        object_in_danger |= torch.abs(object_posistion_in_robot_frame[:, 1]) > y_max
    if z_min is not None:
        object_in_danger |= object_posistion_in_robot_frame[:, 2] < z_min
    if roll_pitch_max is not None:
        object_in_danger |= torch.acos(-object.data.projected_gravity_b[:, 2]).abs() > (roll_pitch_max * math.pi / 180)
    if vel_xy_max is not None:
        object_lin_vel_in_robot_frame = quat_apply_inverse(robot.data.root_quat_w, object.data.root_lin_vel_w - robot.data.root_lin_vel_w)
        object_in_danger |= torch.linalg.norm(object_lin_vel_in_robot_frame[:, :2], dim=1) > vel_xy_max
    return object_in_danger

def object_lose_contact_ngt(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("object_contact_sensor", body_names="Object"),
    ) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name] # type: ignore
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids] # type: ignore
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids] # type: ignore
    return torch.logical_and(last_contact_time > 0.0, current_air_time > 0.0).reshape(-1)

