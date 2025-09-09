from __future__ import annotations
import torch
import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from omni.isaac.lab.utils.math import quat_rotate
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv


def reset_object_state_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    reference_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    reference_asset: RigidObject | Articulation = env.scene[reference_asset_cfg.name]
    # object_states = asset.data.default_root_state[env_ids].clone()
    # write_root_link_pose_to_sim updates root_state_w for articulation but root_link_state_w for rigid objects
    reference_frame_states = reference_asset.data.root_state_w[env_ids].clone()  # root_link_state_w is not updated in 

    object_z = 0.0
    if isinstance(asset.cfg.spawn, sim_utils.CuboidCfg):
        object_z = asset.cfg.spawn.size[2] / 2
    elif isinstance(asset.cfg.spawn, sim_utils.SphereCfg):
        object_z = asset.cfg.spawn.radius
    elif isinstance(asset.cfg.spawn, sim_utils.CylinderCfg) or isinstance(asset.cfg.spawn, sim_utils.CapsuleCfg):
        object_z = asset.cfg.spawn.radius

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
    rand_samples[:, 2] += object_z

    # poses
    positions = reference_frame_states[:, 0:3] + quat_rotate(reference_frame_states[:, 3:7], rand_samples[:, 0:3])
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(reference_frame_states[:, 3:7], orientations_delta)
    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
    velocities = reference_frame_states[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_link_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_com_velocity_to_sim(velocities, env_ids=env_ids)


class ResetObjectStateUniform(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("object"))
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        self.reference_asset_cfg: SceneEntityCfg = cfg.params.get("reference_asset_cfg", SceneEntityCfg("robot"))
        self.reference_asset: RigidObject | Articulation = env.scene[self.reference_asset_cfg.name]
        pose_range = cfg.params.get("pose_range", {})
        pose_range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        self.pose_ranges = torch.tensor(pose_range_list, device=self.asset.device)
        velocity_range = cfg.params.get("velocity_range", {})
        velocity_range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]  
        self.velocity_ranges = torch.tensor(velocity_range_list, device=self.asset.device)

        object_height = []
        for asset_cfg in self.asset.cfg.spawn.assets_cfg:
            if isinstance(asset_cfg, sim_utils.CuboidCfg):
                object_height.append(asset_cfg.size[2] / 2)
            elif isinstance(asset_cfg, sim_utils.SphereCfg):
                object_height.append(asset_cfg.radius)
            elif isinstance(asset_cfg, sim_utils.CylinderCfg) or isinstance(asset_cfg, sim_utils.CapsuleCfg):
                object_height.append(asset_cfg.radius)
        self.object_height = torch.tensor(object_height, device=self.asset.device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        pose_range: dict[str, tuple[float, float]],
        velocity_range: dict[str, tuple[float, float]],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        reference_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ):
        pose_rand_samples = math_utils.sample_uniform(self.pose_ranges[:, 0], self.pose_ranges[:, 1], (len(env_ids), 6), device=self.asset.device)
        reference_frame_states = self.reference_asset.data.root_state_w[env_ids].clone()

        # poses
        positions = reference_frame_states[:, 0:3] + pose_rand_samples[:, 0:3]
        positions[:, 2] +=  self.object_height[env_ids]
        orientations_delta = math_utils.quat_from_euler_xyz(pose_rand_samples[:, 3], pose_rand_samples[:, 4], pose_rand_samples[:, 5])
        orientations = math_utils.quat_mul(reference_frame_states[:, 3:7], orientations_delta)

        # velocities
        velocity_rand_samples = math_utils.sample_uniform(self.velocity_ranges[:, 0], self.velocity_ranges[:, 1], (len(env_ids), 6), device=self.asset.device)
        velocities = reference_frame_states[:, 7:13] + velocity_rand_samples

        # set into the physics simulation
        self.asset.write_root_link_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
        self.asset.write_root_com_velocity_to_sim(velocities, env_ids=env_ids)


class randomize_friction_restitution(ManagerTermBase):
    # based on the original randomize_rigid_body_material but:
    # 1. sample friction and restitution values each time the environment is reset
    # 2. apply the sampled values to all the rigid bodies in the asset_cfg
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # extract the used quantities (to enable type-hinting)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        if not isinstance(self.asset, (RigidObject, Articulation)):
            raise ValueError(
                f"Randomization term 'randomize_rigid_body_material' not supported for asset: '{self.asset_cfg.name}'"
                f" with type: '{type(self.asset)}'."
            )

        # obtain number of shapes per body (needed for indexing the material properties correctly)
        # note: this is a workaround since the Articulation does not provide a direct way to obtain the number of shapes
        #  per body. We use the physics simulation view to obtain the number of shapes per body.
        if isinstance(self.asset, Articulation) and self.asset_cfg.body_ids != slice(None):
            self.num_shapes_per_body = []
            for link_path in self.asset.root_physx_view.link_paths[0]:
                link_physx_view = self.asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
                self.num_shapes_per_body.append(link_physx_view.max_shapes)
            # ensure the parsing is correct
            num_shapes = sum(self.num_shapes_per_body)
            expected_shapes = self.asset.root_physx_view.max_shapes
            if num_shapes != expected_shapes:
                raise ValueError(
                    "Randomization term 'randomize_rigid_body_material' failed to parse the number of shapes per body."
                    f" Expected total shapes: {expected_shapes}, but got: {num_shapes}."
                )
        else:
            # in this case, we don't need to do special indexing
            self.num_shapes_per_body = None

        # obtain parameters for sampling friction and restitution values
        static_friction_range = cfg.params.get("static_friction_range", (1.0, 1.0))
        dynamic_friction_range = cfg.params.get("dynamic_friction_range", (1.0, 1.0))
        restitution_range = cfg.params.get("restitution_range", (0.0, 0.0))

        # sample material properties from the given ranges
        # note: we only sample the materials once during initialization
        #   afterwards these are randomly assigned to the geometries of the asset
        range_list = [static_friction_range, dynamic_friction_range, restitution_range]
        self.ranges = torch.tensor(range_list, device="cpu")

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        static_friction_range: tuple[float, float],
        dynamic_friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        asset_cfg: SceneEntityCfg,
        make_consistent: bool = False,
    ):
        # resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids = env_ids.cpu()

        # retrieve material buffer from the physics simulation
        materials = self.asset.root_physx_view.get_material_properties()
        sampled_materials = torch.rand((len(env_ids), 3), device="cpu") * (self.ranges[:, 1] - self.ranges[:, 0]) + self.ranges[:, 0]
        if make_consistent:
            sampled_materials[:, 1] = torch.min(sampled_materials[:, 0], sampled_materials[:, 1])

        # update material buffer with new samples
        if self.num_shapes_per_body is not None:
            # sample material properties from the given ranges
            for body_id in self.asset_cfg.body_ids:
                # obtain indices of shapes for the body
                start_idx = sum(self.num_shapes_per_body[:body_id])
                end_idx = start_idx + self.num_shapes_per_body[body_id]
                # assign the new materials
                materials[env_ids, start_idx:end_idx] = sampled_materials.unsqueeze(1)
        else:
            # assign all the materials
            materials[env_ids] = sampled_materials[:]

        # apply to simulation
        self.asset.root_physx_view.set_material_properties(materials, env_ids)

