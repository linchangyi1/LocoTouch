from __future__ import annotations
import math
from dataclasses import MISSING
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import SceneEntityCfg, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg, EventTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import locotouch.mdp as mdp


@configclass
class MySceneCfg(InteractiveSceneCfg):
    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    # robots
    robot: ArticulationCfg = MISSING # type: ignore

    # sensors
    robot_contact_senosr = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/(?!sensor.*).*",  # net_history_forces: (N, 3, 17, 3)
        history_length=3,
        track_air_time=True,
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=1000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.6, 0.6, 0.6), intensity=1000.0),
    )


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandGaitLoggingCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 8.0),
        rel_heading_envs=0.0,
        heading_command=False,
        # debug_vis=True,
        ranges=mdp.UniformVelocityCommandGaitLoggingCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.6, 0.6), ang_vel_z=(-math.pi / 2, math.pi / 2)
        ),
        rel_standing_envs=0.1,
    )


@configclass
class ObservationsCfg:
    # observation processing: add noise, clip, scale
    @configclass
    class NoisyProprioceptionCfg(ObservationGroupCfg):
        velocity_commands = ObservationTermCfg(func=mdp.generated_commands, scale=1.0, params={"command_name": "base_velocity"}, history_length=6)
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, scale=0.25, noise=Unoise(n_min=-0.2, n_max=0.2), history_length=6)
        projected_gravity = ObservationTermCfg(func=mdp.projected_gravity, scale=1.0, noise=Unoise(n_min=-0.05, n_max=0.05), history_length=6)
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, scale=1.0, noise=Unoise(n_min=-0.01, n_max=0.01), history_length=6)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5), history_length=6)
        last_action = ObservationTermCfg(func=mdp.last_action, scale=1.0, params={"action_name": "joint_pos"}, history_length=6)  # remember to specify the action name since raw_actions are clipped actions

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class DenoisedProprioceptionCfg(NoisyProprioceptionCfg):
        def __post_init__(self):
            self.enable_corruption = False

    # observation groups
    policy: NoisyProprioceptionCfg = NoisyProprioceptionCfg()
    critic: NoisyProprioceptionCfg = DenoisedProprioceptionCfg()


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionPrevPrevCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,  # don't need to scale again on the raw_actions
        use_default_offset=True,
        clip_raw_actions = True,
        raw_action_clip_value= 100.0,
        raw_action_scale = 0.25,
        )


@configclass
class RewardsCfg:
    # -- alive
    alive = RewardTermCfg(func=mdp.is_alive, weight=10.0)

    # -- velocity tracking (task)
    track_lin_vel_xy = RewardTermCfg(func=mdp.track_lin_vel_xy_pst, weight=1.0, params={"command_name": "base_velocity", "sigma": 0.25})
    track_ang_vel_z = RewardTermCfg(func=mdp.track_ang_vel_z_pst, weight=0.5, params={"command_name": "base_velocity", "sigma": 0.25})

    # -- foot slipping and dragging
    foot_slip = RewardTermCfg(
        func=mdp.foot_slipping_ngt,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot"),
            "sensor_cfg": SceneEntityCfg("robot_contact_senosr", body_names=".*foot"),
            "threshold": 0.5,
        },
    )
    foot_dragging = RewardTermCfg(
        func=mdp.foot_dragging_ngt,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot"),
            "height_threshold": 0.03,
            "foot_vel_xy_threshold": 0.1,
        },
    )

    # -- gait
    gait = RewardTermCfg(
        func=mdp.AdaptiveSymmetricGaitReward,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("robot_contact_senosr"),
            "synced_feet_pair_names": (("a_FR_foot", "d_RL_foot"), ("b_FL_foot", "c_RR_foot")),
            "judge_time_threshold": 1.0e-6,
            "air_time_gait_bound": 0.5,
            "contact_time_gait_bound": 0.5,
            "async_time_tolerance": 0.05,
            "stance_rwd_scale": 1.0,
            "encourage_symmetricity_and_low_frequency": 1.0,  # 0.5 is a threshold for bool check
            "soft_minimum_frequency": 2.0,
            "tolerance_proportion": 0.2,  # (alpha-1.0) in paper
            "rwd_upper_bound": 1.0,
            "rwd_lower_bound": -5.0,
            "vel_tracking_exp_sigma": 0.25,  # the sigma used in velocity tracking reward
            "task_performance_ratio": 1.0,
        },
    )

    # -- regularization
    # - base (torso)
    track_base_height = RewardTermCfg(func=mdp.track_base_height_ngt, weight=-0.5, params={"target_height": 0.27})
    base_z_velocity = RewardTermCfg(func=mdp.base_z_velocity_ngt, weight=-1.0)
    base_roll_pitch_angle = RewardTermCfg(func=mdp.base_roll_pitch_angle_ngt, weight=-1.0)
    base_roll_pitch_velocity = RewardTermCfg(func=mdp.base_roll_pitch_velocity_ngt, weight=-0.2)
    # - joint
    joint_position_limit = RewardTermCfg(func=mdp.joint_position_limit_ngt, weight=-10.0)
    joint_position = RewardTermCfg(
        func=mdp.joint_position_ngt,
        weight=-5.0e-1,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.3,
            })
    joint_acceleration = RewardTermCfg(func=mdp.joint_acceleration_ngt, weight=-5.0e-6)
    joint_velocity = RewardTermCfg(func=mdp.joint_velocity_ngt, weight=-5.0e-3)
    joint_torque = RewardTermCfg(func=mdp.joint_torque_ngt, weight=-2.5e-4)
    action_rate = RewardTermCfg(func=mdp.action_rate_ngt, weight=-0.75)
    # - link
    thigh_calf_collision = RewardTermCfg(
        func=mdp.thigh_calf_collision_ngt,
        weight=-5.,
        params={
            "sensor_cfg": SceneEntityCfg("robot_contact_senosr", body_names=[".*thigh", ".*calf"]),
            "threshold": 0.1,
        },
    )


@configclass
class EventCfg:
    # startup
    randomize_trunk_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
            "mass_distribution_params": (-1, 2),
            "operation": "add",
        },
    )
    randomize_foot_physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot"),
            "static_friction_range": (0.4, 2.0),
            "dynamic_friction_range": (0.4, 2.0),
            "make_consistent": True,
            "restitution_range": (0.0, 0.5),
            "num_buckets": 4000,
        },
    )

    # reset
    reset_base = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.3, 0.3),
                "y": (-0.3, 0.3),
                "z": (-0.05, 0.05),
                "roll": (-math.pi / 6, -math.pi / 6),
                "pitch": (-math.pi / 6, -math.pi / 6),
                "yaw": (-math.pi / 2, -math.pi / 2),
            },
            "velocity_range": {
                "x": (-0.01, 0.01),
                "y": (-0.01, 0.01),
                "z": (0.0, 0.0),
                "roll": (-math.pi / 4, -math.pi / 4),
                "pitch": (-math.pi / 4, -math.pi / 4),
                "yaw": (-math.pi / 2, -math.pi / 2),
            },
        },
    )
    reset_robot_joints = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.03, 0.03),
            "velocity_range": (-0.1, 0.1),
        },
    )

    # interval
    push_robot = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 8.0),
        params={
            "velocity_range": {
                "x": (-1.0, 1.0),
                "y": (-0.6, 0.6),
                "z": (-0.2, 0.2),
                "roll": (-math.pi / 4, -math.pi / 4),
                "pitch": (-math.pi / 4, math.pi / 4),
                "yaw": (-math.pi / 2, math.pi / 2),
                }},
    )


@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    base_orientation = TerminationTermCfg(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": math.pi / 2},
        )
    base_height_below_minimum = TerminationTermCfg(
        func=mdp.root_height_below_minimum,
        params={"asset_cfg": SceneEntityCfg("robot"), "minimum_height": 0.15},
        )
    base_contact = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("robot_contact_senosr", body_names="trunk"), "threshold": 1.0},
        )
    hip_contact = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("robot_contact_senosr", body_names=".*hip"), "threshold": 1.0},
        )


@configclass
class CurriculumCfg:
    velocity_commands = None


@configclass
class LocomotionBaseEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # viewer = ViewerCfg(eye=(2.0, 2.0, 1.0), origin_type="world", env_index=0, asset_name="robot")
    viewer = ViewerCfg(
        eye=(5.0, 5.0, 4.0),
        resolution = (1920, 1080),
        lookat=(-2.0, -2.0, 0.0),
        origin_type="world",
        env_index=0,
        asset_name="robot")
    # Basic settings
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "multiply"
        self.sim.physics_material.restitution_combine_mode = "multiply"
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.robot_contact_senosr is not None:
            self.scene.robot_contact_senosr.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


def smaller_scene_for_playing(env_cfg: LocomotionBaseEnvCfg) -> None:
    env_cfg.scene.num_envs = 50
    env_cfg.scene.env_spacing = 2.5
    env_cfg.sim.physx.gpu_max_rigid_patch_count = 5 * 2**15


@configclass
class LocomotionBaseEnvCfg_PLAY(LocomotionBaseEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        smaller_scene_for_playing(self)


