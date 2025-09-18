from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg, ManagerTermBase, ObservationTermCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_inv, quat_mul, quat_apply_inverse, quat_from_euler_xyz
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


# ----------------- Object State -----------------
def object_state_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("object_contact_sensor", body_names="Object"),
    last_contact_time_threshold: float = 0.00001,
    current_contact_time_threshold: float = 0.00001,
    non_contact_obs: list = [0.0]*13,
    add_uniform_noise: bool = False,
    n_min = -0.03,
    n_max = 0.03,
    scale = 1.0,
    ) -> torch.Tensor:
    # compute the state of the object in the robot frame
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    robot_quat_w = robot.data.root_quat_w
    pos_in_robot_frame = quat_apply_inverse(robot_quat_w, obj.data.root_pos_w - robot.data.root_pos_w)
    lin_vel_in_robot_frame =  quat_apply_inverse(robot_quat_w, obj.data.root_lin_vel_w - robot.data.root_lin_vel_w)
    quat_in_robot_frame = quat_mul(quat_inv(robot_quat_w), obj.data.root_quat_w)
    ang_vel_in_robot_frame = quat_apply_inverse(robot_quat_w, obj.data.root_ang_vel_w - robot.data.root_ang_vel_w)
    state_in_robot_frame = torch.cat([pos_in_robot_frame, lin_vel_in_robot_frame, quat_in_robot_frame, ang_vel_in_robot_frame], dim=-1)

    # compute whether the object have made the first contact
    object_contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    last_contact_time = object_contact_sensor.data.last_contact_time
    current_contact_time = object_contact_sensor.data.current_contact_time
    non_first_contact = torch.logical_and(last_contact_time < last_contact_time_threshold, current_contact_time < current_contact_time_threshold)
    non_contact_env_num = non_first_contact.sum()
    non_first_contact_env_index = non_first_contact.nonzero(as_tuple=True)[0]
    non_contact_obs = torch.tensor(non_contact_obs, device=robot_quat_w.device)
    if non_contact_env_num > 0:
        non_contact_obs = non_contact_obs.repeat(non_contact_env_num, 1)
    if add_uniform_noise:
        euler_angle_min = n_min if isinstance(n_min, float) else torch.tensor(n_min[6:9], device=robot_quat_w.device)
        euler_angle_max = n_max if isinstance(n_max, float) else torch.tensor(n_max[6:9], device=robot_quat_w.device)
        if not isinstance(n_min, float):
            n_min = torch.tensor(n_min[0:6]+[0.0]*4+n_min[9::], device=robot_quat_w.device)  # mask euler angles
            n_max = torch.tensor(n_max[0:6]+[0.0]*4+n_max[9::], device=robot_quat_w.device)  # mask euler angles
        state_in_robot_frame += torch.rand_like(state_in_robot_frame) * (n_max - n_min) + n_min
        delta_euler_xyz = torch.rand(size=(robot_quat_w.shape[0], 3), device=robot_quat_w.device) * (euler_angle_max - euler_angle_min) + euler_angle_min
        noisy_quat = quat_from_euler_xyz(delta_euler_xyz[:, 0], delta_euler_xyz[:, 1], delta_euler_xyz[:, 2])
        state_in_robot_frame[:, 6:10] = quat_mul(state_in_robot_frame[:, 6:10], noisy_quat)
        if non_contact_env_num > 0:
            non_contact_obs += torch.rand_like(non_contact_obs) * (n_max - n_min) + n_min
            non_contact_obs[:, 6:10] = quat_mul(non_contact_obs[:, 6:10], noisy_quat[non_first_contact_env_index])

    scale = scale if isinstance(scale, float) else torch.tensor(scale, device=robot_quat_w.device)
    state_in_robot_frame *= scale
    if non_contact_env_num > 0:
        non_contact_obs *= scale
        state_in_robot_frame[non_first_contact_env_index] = non_contact_obs

    return state_in_robot_frame


# ----------------- Tactile Signals -----------------
class TactileSignals(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg")
        self.sensor_cfg: SceneEntityCfg = cfg.params.get("sensor_cfg")
        self.asset: RigidObject = env.scene[self.asset_cfg.name]
        self.contact_sensor: ContactSensor = env.scene.sensors[self.sensor_cfg.name]
        tactile_signal_shape: tuple = cfg.params.get("tactile_signal_shape")
        self.tactile_signals_shape = (self.num_envs, tactile_signal_shape[0], tactile_signal_shape[1])

        # Original signals
        self.original_contact_taxels: torch.Tensor = torch.zeros(self.tactile_signals_shape, device=self.asset.device, dtype=torch.bool)
        self.original_normal_forces: torch.Tensor = torch.zeros_like(self.original_contact_taxels, dtype=torch.float)
        self.original_normalized_forces: torch.Tensor = torch.zeros_like(self.original_contact_taxels, dtype=torch.float)
        self.original_min_max_normalized_signals: torch.Tensor = torch.zeros_like(self.original_contact_taxels, dtype=torch.float)
        self.original_discretized_signals: torch.Tensor = torch.zeros_like(self.original_contact_taxels, dtype=torch.float)

        # Processed signals
        self.processed_contact_taxels: torch.Tensor = torch.zeros_like(self.original_contact_taxels, dtype=torch.bool)
        self.processed_normal_forces: torch.Tensor = torch.zeros_like(self.original_contact_taxels, dtype=torch.float)
        self.processed_normalized_forces: torch.Tensor = torch.zeros_like(self.original_contact_taxels, dtype=torch.float)
        self.processed_min_max_normalized_signals: torch.Tensor = torch.zeros_like(self.original_contact_taxels, dtype=torch.float)
        self.processed_discretized_signals: torch.Tensor = torch.zeros_like(self.original_contact_taxels, dtype=torch.float)

        # Contact Taxels
        self.contact_threshold: float = cfg.params.get("contact_threshold")
        self.contact_threshold_envs_sensors: torch.Tensor = torch.ones(self.tactile_signals_shape, device=self.asset.device) * self.contact_threshold
        self.add_threshold_noise: bool = cfg.params.get("add_threshold_noise")
        if self.add_threshold_noise:
            self.threshold_n_min: float = cfg.params.get("threshold_n_min")
            self.threshold_n_max: float = cfg.params.get("threshold_n_max")
            self.contact_threshold_envs_sensors = self.contact_threshold + torch.rand_like(self.contact_threshold_envs_sensors) * (self.threshold_n_max - self.threshold_n_min) + self.threshold_n_min
        self.contact_dropout_prob: float = cfg.params.get("contact_dropout_prob")
        self.contact_addition_prob: float = cfg.params.get("contact_addition_prob")
        self.add_continuous_artifact: bool = cfg.params.get("add_continuous_artifact") > 0.5
        if self.add_continuous_artifact:
            self.artifact_taxel_num_min: int = cfg.params.get("artifact_taxel_num_min")
            self.artifact_taxel_num_max: int = cfg.params.get("artifact_taxel_num_max")
            self.artifact_num = torch.randint(
                low=self.artifact_taxel_num_min,
                high=self.artifact_taxel_num_max + 1,  # +1 because upper bound is exclusive
                size=(N,),
                device=self.asset.device
            )

        # Normal Forces
        self.add_force_noise: bool = cfg.params.get("add_force_noise")
        if self.add_force_noise:
            self.force_n_prop_min: float = cfg.params.get("force_n_prop_min")
            self.force_n_prop_max: float = cfg.params.get("force_n_prop_max")
        self.maximal_force: float = cfg.params.get("maximal_force")

        # Discretization
        self.total_levels: int = cfg.params.get("total_levels")
        self.add_level_noise: bool = cfg.params.get("add_level_noise")
        if self.add_level_noise:
            self.level_n_min: float = cfg.params.get("level_n_min")
            self.level_n_max: float = cfg.params.get("level_n_max")

    def get_original_signals(self):
        # get the normal forces in local sensor frame
        self.original_normal_forces[:] = -quat_apply_inverse(
            self.asset.data.body_quat_w[:, self.asset_cfg.body_ids][:, :, :],
            self.contact_sensor.data.net_forces_w[:, self.sensor_cfg.body_ids])[..., 2].reshape(self.tactile_signals_shape)
        self.original_contact_taxels[:] = self.original_normal_forces > self.contact_threshold_envs_sensors

    def process_original_signals(self):
        self.original_normalized_forces[:] = torch.clamp(self.original_normal_forces / self.maximal_force, 0.0, 1.0)
        _, self.original_min_max_normalized_signals[:] = self.compute_min_max_normalized_signals(self.original_contact_taxels, self.original_normalized_forces)
        _, self.original_discretized_signals[:] = self.compute_discretized_signals(self.original_contact_taxels, self.original_min_max_normalized_signals)

    def get_normal_forces(self):
        self.get_original_signals()
        contact_taxels, normal_forces = self.original_contact_taxels.clone(), self.original_normal_forces.clone()

        # apply contact dropout:
        # dropout some contact taxels, make their forces to be [0, contact_threshold]
        if self.contact_dropout_prob > 0.0:
            dropout_mask = torch.rand_like(contact_taxels.float()) < self.contact_dropout_prob
            dropout_taxels = torch.logical_and(contact_taxels, dropout_mask)
            normal_forces[dropout_taxels] = torch.rand_like(normal_forces[dropout_taxels]) * self.contact_threshold_envs_sensors[dropout_taxels]
            contact_taxels[dropout_taxels] = False

        # apply contact addition:
        # add some non contact taxels, make their forces to be [contact_threshold, 1.2*contact_threshold]
        if self.contact_addition_prob > 0.0:
            addition_mask = torch.rand_like(contact_taxels.float()) < self.contact_addition_prob
            addition_taxels = torch.logical_and(~contact_taxels, addition_mask)
            # print(f"addition_taxels: {addition_taxels.sum()}")
            normal_forces[addition_taxels] = self.contact_threshold_envs_sensors[addition_taxels] * (1.0 + 0.2*torch.rand_like(normal_forces[addition_taxels]))
            contact_taxels[addition_taxels] = True

        # apply force noise
        if self.add_force_noise:
            normal_forces[contact_taxels] *= 1.0 + (torch.rand_like(normal_forces[contact_taxels]) * (self.force_n_prop_max - self.force_n_prop_min) + self.force_n_prop_min)
            normal_forces = torch.clamp(normal_forces, min=0.0)

            too_small_forces = torch.logical_and(contact_taxels, normal_forces < self.contact_threshold_envs_sensors)
            normal_forces[too_small_forces] = self.contact_threshold_envs_sensors[too_small_forces] * (1.0 + 0.2*torch.rand_like(normal_forces[too_small_forces]))

        # save to processed signals
        self.processed_contact_taxels = contact_taxels.clone()
        self.processed_normal_forces = normal_forces

        return contact_taxels, normal_forces

    def get_normalized_forces(self):
        contact_taxels, normal_forces = self.get_normal_forces()
        normalized_forces = torch.clamp(normal_forces / self.maximal_force, 0.0, 1.0)
        self.processed_normalized_forces = normalized_forces.clone()
        return contact_taxels, normalized_forces

    def compute_min_max_normalized_signals(self, contact_taxels, normalized_forces):
        valid_normalized_forces = torch.where(contact_taxels, normalized_forces, torch.zeros_like(normalized_forces))
        # Get the min and max forces for each env
        flat_forces = valid_normalized_forces.view(valid_normalized_forces.shape[0], -1)
        min_forces = flat_forces.min(dim=-1, keepdim=True)[0].unsqueeze(-1)  # (N, 1, 1)
        max_forces = flat_forces.max(dim=-1, keepdim=True)[0].unsqueeze(-1)  # (N, 1, 1)

        # Compute range and avoid division by zero
        force_range = torch.where((max_forces - min_forces) > 0.0,
                                max_forces - min_forces,
                                torch.ones_like(max_forces))  # (N, 1, 1)

        # Normalize and clamp
        normalized_signals = (valid_normalized_forces - min_forces) / force_range
        normalized_signals = torch.clamp(normalized_signals, 0.0, 1.0)
        return contact_taxels, normalized_signals

    def compute_discretized_signals(self, contact_taxels, normalized_signals):
        # Uniform Discretization
        discrete_bin = 1.0 / self.total_levels
        discretized_signals = torch.round(normalized_signals / discrete_bin)
        if self.add_level_noise:
            level_noise = torch.rand_like(discretized_signals) * (self.level_n_max - self.level_n_min) + self.level_n_min
            discretized_signals = discretized_signals + level_noise
        discretized_signals *= discrete_bin
        discretized_signals = torch.clamp(discretized_signals, 0.0, 1.0)
        discretized_signals[~contact_taxels] = 0.0
        return contact_taxels, discretized_signals

    def get_min_max_normalized_signals(self):
        contact_taxels, normalized_forces = self.get_normalized_forces()
        contact_taxels, min_max_normalized_signals = self.compute_min_max_normalized_signals(contact_taxels, normalized_forces)
        self.processed_min_max_normalized_signals = min_max_normalized_signals.clone()
        return contact_taxels, min_max_normalized_signals

    def get_discretized_signals(self):
        contact_taxels, min_max_normalized_signals = self.get_min_max_normalized_signals()
        contact_taxels, discretized_signals = self.compute_discretized_signals(contact_taxels, min_max_normalized_signals)
        self.processed_discretized_signals = discretized_signals.clone()
        return contact_taxels, discretized_signals

    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="sensor_.*"),
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("tactile_contact_sensor", body_names="sensor_.*"),
        tactile_signal_shape: tuple = (17, 13),
        contact_threshold = 0.1,
        add_threshold_noise: bool = False,
        threshold_n_min = -0.03,
        threshold_n_max = 0.03,
        contact_dropout_prob: float = 0.0,
        contact_addition_prob: float = 0.0,
        add_continuous_artifact: float = 0.0,
        artifact_taxel_num_min: int = 0,
        artifact_taxel_num_max: int = 3,
        add_force_noise: bool = False,
        force_n_prop_min = -0.03,
        force_n_prop_max = 0.03,
        maximal_force: float = 1.0,
        total_levels: int = 20,
        add_level_noise: bool = False,
        level_n_min = -3,
        level_n_max: float = 3,
        ) -> torch.Tensor:
        self.get_original_signals()
        self.process_original_signals()
        return torch.stack([
            self.original_contact_taxels.float(),
            self.original_normalized_forces,
            self.original_min_max_normalized_signals,
            self.original_discretized_signals], dim=1).flatten(start_dim=1)


class BinaryTactileSignals(TactileSignals):
    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="sensor_.*"),
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("tactile_contact_sensor", body_names="sensor_.*"),
        tactile_signal_shape: tuple = (17, 13),
        contact_threshold = 0.1,
        add_threshold_noise: bool = False,
        threshold_n_min = -0.03,
        threshold_n_max = 0.03,
        contact_dropout_prob: float = 0.0,
        contact_addition_prob: float = 0.0,
        add_continuous_artifact: float = 0.0,
        artifact_taxel_num_min: int = 0,
        artifact_taxel_num_max: int = 3,
        add_force_noise: bool = False,
        force_n_prop_min = -0.03,
        force_n_prop_max = 0.03,
        maximal_force: float = 1.0,
        total_levels: int = 20,
        add_level_noise: bool = False,
        level_n_min = -3,
        level_n_max: float = 3,
        ) -> torch.Tensor:
        contact_taxels, _ = self.get_normal_forces()
        # why two channels of binary maps? --> we wanted to keep one channel with contact_taxels and the other channel with different tactile signals (Though this is not reported in the paper)
        return torch.stack([contact_taxels.float(), contact_taxels.float()], dim=1).flatten(start_dim=1)


class NormalizedTactileSignals(BinaryTactileSignals):
    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="sensor_.*"),
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("tactile_contact_sensor", body_names="sensor_.*"),
        tactile_signal_shape: tuple = (17, 13),
        contact_threshold = 0.1,
        add_threshold_noise: bool = False,
        threshold_n_min = -0.03,
        threshold_n_max = 0.03,
        contact_dropout_prob: float = 0.0,
        contact_addition_prob: float = 0.0,
        add_continuous_artifact: float = 0.0,
        artifact_taxel_num_min: int = 0,
        artifact_taxel_num_max: int = 3,
        add_force_noise: bool = False,
        force_n_prop_min = -0.03,
        force_n_prop_max = 0.03,
        maximal_force: float = 1.0,
        total_levels: int = 20,
        add_level_noise: bool = False,
        level_n_min = -3,
        level_n_max: float = 3,
        ) -> torch.Tensor:
        contact_taxels, normalized_signals = self.get_min_max_normalized_signals()
        return torch.stack([contact_taxels.float(), normalized_signals], dim=1).flatten(start_dim=1)


class DiscreteTactileSignals(NormalizedTactileSignals):
    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="sensor_.*"),
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("tactile_contact_sensor", body_names="sensor_.*"),
        tactile_signal_shape: tuple = (17, 13),
        contact_threshold = 0.1,
        add_threshold_noise: bool = False,
        threshold_n_min = -0.03,
        threshold_n_max = 0.03,
        contact_dropout_prob: float = 0.0,
        contact_addition_prob: float = 0.0,
        add_continuous_artifact: float = 0.0,
        artifact_taxel_num_min: int = 0,
        artifact_taxel_num_max: int = 3,
        add_force_noise: bool = False,
        force_n_prop_min = -0.03,
        force_n_prop_max = 0.03,
        maximal_force: float = 1.0,
        total_levels: int = 20,
        add_level_noise: bool = False,
        level_n_min = -3,
        level_n_max: float = 3,
        ) -> torch.Tensor:
        contact_taxels, discretized_signals = self.get_discretized_signals()
        return torch.stack([contact_taxels.float(), discretized_signals], dim=1).flatten(start_dim=1)


class CotinuousTactileSignals(TactileSignals):
    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="sensor_.*"),
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("tactile_contact_sensor", body_names="sensor_.*"),
        tactile_signal_shape: tuple = (17, 13),
        contact_threshold = 0.1,
        add_threshold_noise: bool = False,
        threshold_n_min = -0.03,
        threshold_n_max = 0.03,
        contact_dropout_prob: float = 0.0,
        contact_addition_prob: float = 0.0,
        add_continuous_artifact: float = 0.0,
        artifact_taxel_num_min: int = 0,
        artifact_taxel_num_max: int = 3,
        add_force_noise: bool = False,
        force_n_prop_min = -0.03,
        force_n_prop_max = 0.03,
        maximal_force: float = 1.0,
        total_levels: int = 20,
        add_level_noise: bool = False,
        level_n_min = -3,
        level_n_max: float = 3,
        ) -> torch.Tensor:
        contact_taxels, normalized_forces = self.get_normalized_forces()
        return torch.stack([contact_taxels.float(), normalized_forces], dim=1).flatten(start_dim=1)
        # return normalized_forces


class ProcessedTactileSignals(TactileSignals):
    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="sensor_.*"),
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("tactile_contact_sensor", body_names="sensor_.*"),
        tactile_signal_shape: tuple = (17, 13),
        contact_threshold = 0.1,
        add_threshold_noise: bool = False,
        threshold_n_min = -0.03,
        threshold_n_max = 0.03,
        contact_dropout_prob: float = 0.0,
        contact_addition_prob: float = 0.0,
        add_continuous_artifact: float = 0.0,
        artifact_taxel_num_min: int = 0,
        artifact_taxel_num_max: int = 3,
        add_force_noise: bool = False,
        force_n_prop_min = -0.03,
        force_n_prop_max = 0.03,
        maximal_force: float = 1.0,
        total_levels: int = 20,
        add_level_noise: bool = False,
        level_n_min = -3,
        level_n_max: float = 3,
        ) -> torch.Tensor:
        contact_taxels, _ = self.get_discretized_signals()
        return torch.stack([
            self.processed_contact_taxels.float(),
            self.processed_normalized_forces,
            self.processed_min_max_normalized_signals,
            self.processed_discretized_signals], dim=1).flatten(start_dim=1)


