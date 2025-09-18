import math
import isaaclab.sim as sim_utils
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass

import locotouch.mdp as mdp
from .object_transport_teacher_env_cfg import *


@configclass
class CylinderTransportTeacherEnvCfg(ObjectTransportTeacherEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # add cylinders to the scene
        self.scene.object.spawn=sim_utils.CylinderCfg(
            radius=0.05,
            height=0.3,
            axis="Y",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            activate_contact_sensors=True,
            # collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=1.0e-9, rest_offset=-0.002),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 0.0), opacity=1.0),
        )

        # vel curiculum
        self.curriculum.velocity_commands.params["error_threshold_lin"] = 0.07
        self.curriculum.velocity_commands.params["error_threshold_ang"] = 0.08

        # revise rewards, events, and termination for cylinders
        self.rewards.object_roll_pitch_angle.func = mdp.object_relative_roll_angle_ngt
        self.rewards.object_roll_pitch_angle.weight = -0.05
        self.rewards.object_roll_pitch_velocity.func = mdp.object_relative_roll_velocity_ngt
        self.rewards.object_roll_pitch_velocity.weight = -0.05
        self.rewards.object_yaw_alignment.weight = -0.1
        self.events.reset_object_position.params["pose_range"]["pitch"] = (-math.pi, math.pi)
        self.terminations.object_bad_orientation = TerminationTermCfg(
            func=mdp.bad_roll,
            params={"asset_cfg": SceneEntityCfg("object"),
                    "limit_angle": math.pi / 3,
                    },
            )

@configclass
class CylinderTransportTeacherEnvCfg_PLAY(CylinderTransportTeacherEnvCfg):
    def __post_init__(self) -> None:
        self.scene.num_envs = 20
        super().__post_init__()
        locotouch_object_transport_play_env_post_init_func(self)

