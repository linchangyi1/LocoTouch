import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.utils import configclass

import numpy as np
import locotouch.mdp as mdp
from locotouch.assets.locotouch import LocoTouch_Without_Tactile_CFG
from .object_transport_teacher_env_cfg import locotouch_object_transport_play_env_post_init_func
from .cylinder_transport_teacher_env_cfg import CylinderTransportTeacherEnvCfg


@configclass
class RandCylinderTransportTeacherEnvCfg(CylinderTransportTeacherEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # disable 'replicate_physics' to perform 'modify_collision_properties'
        self.scene.replicate_physics = False
        self.scene.robot = LocoTouch_Without_Tactile_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # add cylinders with different properties
        env_num = self.scene.num_envs
        # env_num = 2000
        radius_range = (0.03, 0.07)
        height_range = (0.1, 0.4)
        size_range = np.array([radius_range, height_range])
        size_samples = np.random.uniform(size_range[:, 0], size_range[:, 1], (env_num, 2))
        color_samples = np.random.uniform(0.0, 1.0, (env_num, 3)).astype(np.float32)
        self.scene.object = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Object",
            spawn=sim_utils.MultiAssetSpawnerCfg(
                assets_cfg=[
                    sim_utils.CylinderCfg(
                        radius=float(size_samples[i, 0]),
                        height=float(size_samples[i, 1]),
                        axis="Y",
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(map(float, color_samples[i]))),) # type: ignore
                    for i in range(env_num) ],
                random_choice=False,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,),
                activate_contact_sensors=True,
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=1.0e-9, rest_offset=-0.002),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
            )
        self.events.reset_object_position.func = mdp.ResetObjectStateUniform

        # use easier randomization for randomized cylinders
        self.events.randomize_object_physics_material.params["static_friction_range"] = (0.3, 1.0)
        self.events.randomize_trunk_sensor_physics_material.params["static_friction_range"] = (0.3, 1.0)

        # vel curiculum params
        self.curriculum.velocity_commands.params["reset_envs_episode_length"] = 0.98
        self.curriculum.velocity_commands.params["error_threshold_lin"] = 0.08
        self.curriculum.velocity_commands.params["error_threshold_ang"] = 0.1


class RandCylinderTransportTeacherEnvCfg_PLAY(RandCylinderTransportTeacherEnvCfg):
    def __post_init__(self) -> None:
        self.scene.num_envs = 20
        super().__post_init__()
        locotouch_object_transport_play_env_post_init_func(self)

