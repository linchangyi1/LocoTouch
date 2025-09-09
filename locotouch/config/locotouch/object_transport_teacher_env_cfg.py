import math
from omni.isaac.lab.managers import SceneEntityCfg, ObservationGroupCfg, ObservationTermCfg, TerminationTermCfg, EventTermCfg, RewardTermCfg
from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.sensors import ContactSensorCfg

import locotouch.mdp as mdp
from locotouch.config.base.locomotion_base_env_cfg import ObservationsCfg
from .locomotion_vel_cur_env_cfg import LocomotionVelCurEnvCfg, locotouch_locomotion_vel_cur_play_env_post_init_func


@configclass
class NoisyObjectStateCfg(ObservationGroupCfg):
    object_state = ObservationTermCfg(
        func=mdp.object_state_in_robot_frame,
        scale=1.0,
        history_length=6,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "object_cfg": SceneEntityCfg("object"),
            "sensor_cfg": SceneEntityCfg("object_contact_sensor", body_names="Object"),
            "last_contact_time_threshold": 0.00000001,
            "current_contact_time_threshold": 0.00000001,  # at the N-th step's contact, the robot can sense the object state
            "non_contact_obs": [0.0]*3 + [0.0]*3 + [1.0] +[0.0]*3 + [0.0]*3,
            "add_uniform_noise": True,
            "n_min": [-0.01, -0.01, -0.005] + [-0.2]*3 + [-0.05]*3 + [-0.2]*3,  # euler angles noise although obs is quaternion
            "n_max": [0.01, 0.01, 0.005] + [0.2]*3 + [0.05]*3 + [0.2]*3,
            "scale": [1.0]*3 + [0.5]*3 + [1.0]*4 + [0.25]*3,
        },
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class NoisyProprioceptionObjectStateCfg(ObservationsCfg.NoisyProprioceptionCfg):
    def __post_init__(self):
        super().__post_init__()
        object_state_cfg = NoisyObjectStateCfg()
        self.object_state = object_state_cfg.object_state
        self.history_length = 6


@configclass
class DenoisedProprioceptionObjectStateCfg(NoisyProprioceptionObjectStateCfg):
    def __post_init__(self):
        super().__post_init__()
        self.enable_corruption = False
        self.object_state.params["add_uniform_noise"] = False


@configclass
class   ObjectTransportTeacherEnvCfg(LocomotionVelCurEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # increase the rigid patch count for more objects
        self.sim.physx.gpu_max_rigid_patch_count = 4096 * 4096

        # add objects and their contact sensors
        self.scene.object = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Object",
            spawn=None,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        )
        self.scene.object_contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Object",  # net_history_forces: (N, 3, 1, 3)
            history_length=3,
            track_air_time=True,
        )

        # commands sampling settings
        self.commands.base_velocity.rel_standing_envs=0.1  # make sure zero commands are sampled
        self.commands.base_velocity.final_rel_standing_envs = 0.05  # use a smaller value due to 'final_initial_zero_command_steps'
        self.commands.base_velocity.initial_zero_command_steps = 0  # set to 0 to encourage exploration
        self.commands.base_velocity.final_initial_zero_command_steps = 50  # set to 50 to ensure the object stabilizes on the robot

        # smaller velocity commands for object transport
        self.curriculum.velocity_commands.params["command_maximum_ranges"] = [0.5, 0.25, math.pi / 4]  # v_x, v_y, w_z

        # observations
        self.observations.policy = NoisyProprioceptionObjectStateCfg()
        self.observations.critic = DenoisedProprioceptionObjectStateCfg()

        # rewards
        self.rewards.gait.func = mdp.AdaptiveSymmetricGaitRewardwithObject
        self.rewards.object_xy_position = RewardTermCfg(func=mdp.object_relative_xy_position_ngt, weight=-50.0,
                                                          params={"work_only_when_cmd": 1})
        self.rewards.object_xy_velocity = RewardTermCfg(func=mdp.object_relative_xy_velocity_ngt, weight=0.0)
        self.rewards.object_z_contact = RewardTermCfg(func=mdp.object_lose_contact_ngt, weight=0.0)
        self.rewards.object_z_velocity = RewardTermCfg(func=mdp.object_relative_z_velocity_ngt, weight=-0.5)
        self.rewards.object_roll_pitch_angle = RewardTermCfg(func=mdp.object_relative_roll_pitch_angle_ngt, weight=0.0)
        self.rewards.object_roll_pitch_velocity = RewardTermCfg(func=mdp.object_relative_roll_pitch_velocity_ngt, weight=0.0)
        self.rewards.object_yaw_alignment = RewardTermCfg(func=mdp.object_relative_yaw_angle_ngt, weight=0.0,
                                                          params={"work_only_when_cmd": 1})
        self.rewards.object_dangerous_state = RewardTermCfg(func=mdp.object_dangerous_state_ngt, weight=-50.0,
                                                    params={"robot_cfg": SceneEntityCfg("robot"),
                                                            "object_cfg": SceneEntityCfg("object"),
                                                            "x_max": 0.125,
                                                            "y_max": 0.097,
                                                            "z_min": 0.095,
                                                            "roll_pitch_max": None,  # in degree
                                                            "vel_xy_max": 2.5,})

        # termination
        self.terminations.base_contact = None
        self.terminations.object_below_robot = TerminationTermCfg(
            func=mdp.object_below_robot,
            params={"robot_cfg": SceneEntityCfg("robot"),
                    "object_cfg": SceneEntityCfg("object"),
                    },
            )

        # events: friction, pose, mass, perturbation, etc.
        self.events.randomize_foot_physics_material.params["static_friction_range"] = (0.6, 1.5)
        self.events.randomize_foot_physics_material.params["dynamic_friction_range"] = (0.6, 1.5)
        self.events.randomize_foot_physics_material.params["restitution_range"] = (0.0, 0.3)
        self.events.randomize_trunk_sensor_physics_material = EventTermCfg(
            func=mdp.randomize_friction_restitution,  # this will make the friction of the bodies indicated consistent
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
                "static_friction_range": (0.1, 0.8),
                "dynamic_friction_range": (1.0, 1.0),
                "make_consistent": True,
                "restitution_range": (0.0, 0.2),
            },
        )
        self.events.randomize_object_physics_material = EventTermCfg(
            func=mdp.randomize_rigid_body_material,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("object", body_names="Object"),
                "static_friction_range": (0.1, 0.8),
                "dynamic_friction_range": (1.0, 1.0),
                "make_consistent": True,
                "restitution_range": (0.0, 0.2),
                "num_buckets": 8000,
            },
        )
        self.events.reset_base.params ={
                "pose_range": {
                    "x": (-0.3, 0.3),
                    "y": (-0.3, 0.3),
                    "z": (0.0, 0.0),
                    "roll": (-0.0, 0.0),
                    "pitch": (-0.0, 0.0),
                    "yaw": (-0.0, 0.0),
                },
                "velocity_range": {
                    "x": (-0.01, 0.01),
                    "y": (-0.01, 0.01),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                }}
        self.events.reset_object_position = EventTermCfg(
            func=mdp.reset_object_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.05, 0.05),
                    "y": (-0.04, 0.04),
                    "z": (0.095, 0.10),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (-math.pi / 6, math.pi / 6)
                    },
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object", body_names="Object"),
                "reference_asset_cfg": SceneEntityCfg("robot"),
            },
            )
        self.events.object_mass_randomization = EventTermCfg(
            func=mdp.randomize_rigid_body_mass,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("object", body_names="Object"),
                "mass_distribution_params": (-0.5, 1.5),
                "operation": "add",
                "distribution": "uniform",
            },
        )
        self.events.push_robot.interval_range_s = (6.0, 10.0)
        self.events.push_robot.params["velocity_range"]["x"] = (-0.4, 0.4)
        self.events.push_robot.params["velocity_range"]["y"] = (-0.3, 0.3)
        self.events.push_robot.params["velocity_range"]["z"] = (-0.1, 0.1)
        # self.events.push_robot.params["velocity_range"]["roll"] = (-math.pi / 20, math.pi / 20)
        # self.events.push_robot.params["velocity_range"]["pitch"] = (-math.pi / 20, math.pi / 20)
        # self.events.push_robot.params["velocity_range"]["yaw"] = (-math.pi / 5, math.pi / 5)
        self.events.push_robot.params["velocity_range"]["roll"] = (0, 0)
        self.events.push_robot.params["velocity_range"]["pitch"] = (0, 0)
        self.events.push_robot.params["velocity_range"]["yaw"] = (0, 0)
        self.events.push_object = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(6.0, 8.0),
            params={
                "asset_cfg": SceneEntityCfg("object", body_names="Object"),
                "velocity_range": {
                    "x": (-0.3, 0.3), "y": (-0.3, 0.3), "z": (-0.2, 0.2),
                    "roll": (-math.pi / 20, math.pi / 20), "pitch": (-math.pi / 20, math.pi / 20), "yaw": (-math.pi / 5, math.pi / 5),},
                },
        )

def locotouch_object_transport_play_env_post_init_func(env_cfg: ObjectTransportTeacherEnvCfg) -> None:
    locotouch_locomotion_vel_cur_play_env_post_init_func(env_cfg)

    # for debugging
    # env_cfg.scene.robot_contact_senosr.debug_vis = True
    # env_cfg.commands.base_velocity.binary_maximal_command = True
    # env_cfg.episode_length_s = 4.0
    # env_cfg.events.reset_object_position.params["pose_range"]["x"] = (0.05, 0.05)
    # env_cfg.events.reset_object_position.params["pose_range"]["x"] = (-0.05, -0.05)
    # env_cfg.events.reset_object_position.params["pose_range"]["y"] = (0.04, 0.04)
    # env_cfg.events.reset_object_position.params["pose_range"]["yaw"] = (-0.1, -0.1)
    # env_cfg.events.push_robot.interval_range_s = (0.5, 0.5)
    # env_cfg.events.push_robot.params["velocity_range"]["x"] = (0.6, 0.6)
    # env_cfg.events.push_robot.params["velocity_range"]["y"] = (0.0, 0.0)
    # env_cfg.events.push_object.interval_range_s = (0.5, 0.5)
    # env_cfg.events.push_object.interval_range_s = (2.0, 2.0)
    # env_cfg.events.push_object.params["velocity_range"]["x"] = (-0.4, 0.4)
    # env_cfg.events.push_robot.params["velocity_range"]["y"] = (-0.3, 0.3)

# this function is used to set the full velocity range for resuming training (understand it before using it)
def use_full_vel_range_for_resume_training(env_cfg: ObjectTransportTeacherEnvCfg) -> None:
    com_max_ranges = env_cfg.curriculum.velocity_commands.params["command_maximum_ranges"]
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (-com_max_ranges[0], com_max_ranges[0])
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (-com_max_ranges[1], com_max_ranges[1])
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (-com_max_ranges[2], com_max_ranges[2])
    env_cfg.commands.base_velocity.initial_zero_command_steps = env_cfg.commands.base_velocity.final_initial_zero_command_steps
    env_cfg.commands.base_velocity.rel_standing_envs = env_cfg.commands.base_velocity.final_rel_standing_envs


