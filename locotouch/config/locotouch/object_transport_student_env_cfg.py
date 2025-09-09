from omni.isaac.lab.utils import configclass
from omni.isaac.lab.sensors import ContactSensorCfg
from omni.isaac.lab.managers import SceneEntityCfg, ObservationGroupCfg, ObservationTermCfg
import locotouch.mdp as mdp
from locotouch.config.base.locomotion_base_env_cfg import ObservationsCfg
from locotouch.assets.locotouch import LocoTouch_CFG
from .object_transport_teacher_env_cfg import ObjectTransportTeacherEnvCfg, NoisyObjectStateCfg
from .rand_cylinder_transport_teacher_env_cfg import RandCylinderTransportTeacherEnvCfg


# ------------------------ Tactile Obsevation ------------------------
@configclass
class NoisyTactileCfg(ObservationGroupCfg):
    tactile_signals = ObservationTermCfg(
        func=mdp.TactileSignals,
        params={
        "asset_cfg": SceneEntityCfg("robot", body_names="sensor_.*"),
        "sensor_cfg": SceneEntityCfg("tactile_contact_sensor", body_names="sensor_.*"),
        "tactile_signal_shape": (17, 13),
        "contact_threshold": 0.05,
        "add_threshold_noise": True,
        "threshold_n_min": -0.05 * 0.2,
        "threshold_n_max": 0.05 * 0.2,
        "contact_dropout_prob": 0.005,
        "contact_addition_prob": 0.005,
        "add_continuous_artifact": 0.0,
        "artifact_taxel_num_min": 0,
        "artifact_taxel_num_max": 3,
        "add_force_noise": True,
        "force_n_prop_min": -0.1,
        "force_n_prop_max": 0.1,
        "maximal_force": 3.0,
        "total_levels": 5,
        "add_level_noise": True,
        "level_n_min": -1,
        "level_n_max": 1,
        },
        scale=1.0,
        )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


# Though we provide other formats here, only binary maps are evaluated in our project.
# ----- Binary-Tactile
@configclass
class NoisyBinaryTactileCfg(NoisyTactileCfg):
    def __post_init__(self):
        super().__post_init__()
        self.tactile_signals.func = mdp.BinaryTactileSignals

@configclass
class NoisyProprioceptionBinaryTactileCfg(ObservationsCfg.NoisyProprioceptionCfg):
    def __post_init__(self):
        super().__post_init__()
        tactile_input_cfg = NoisyBinaryTactileCfg()
        self.tactile_input = tactile_input_cfg.tactile_signals

@configclass
class DenoisedProprioceptionBinaryTactileCfg(NoisyProprioceptionBinaryTactileCfg):
    def __post_init__(self):
        super().__post_init__()
        self.enable_corruption = False
        self.tactile_input.params["add_threshold_noise"] = False
        self.tactile_input.params["add_force_noise"] = False
        self.tactile_input.params["add_level_noise"] = False

# ----- Normalized-Tactile
@configclass
class NoisyNormalizedTactileCfg(NoisyTactileCfg):
    def __post_init__(self):
        super().__post_init__()
        self.tactile_signals.func = mdp.NormalizedTactileSignals


@configclass
class NoisyProprioceptionNormalizedTactileCfg(ObservationsCfg.NoisyProprioceptionCfg):
    def __post_init__(self):
        super().__post_init__()
        tactile_input_cfg = NoisyNormalizedTactileCfg()
        self.tactile_input = tactile_input_cfg.tactile_signals

@configclass
class DenoisedProprioceptionNormalizedTactileCfg(NoisyProprioceptionNormalizedTactileCfg):
    def __post_init__(self):
        super().__post_init__()
        self.enable_corruption = False
        self.tactile_input.params["add_threshold_noise"] = False
        self.tactile_input.params["add_force_noise"] = False
        self.tactile_input.params["add_level_noise"] = False

# ----- Discretized-Tactile
@configclass
class NoisyDiscretizedTactileCfg(NoisyTactileCfg):
    def __post_init__(self):
        super().__post_init__()
        self.tactile_signals.func = mdp.DiscreteTactileSignals
        self.tactile_signals.params["add_force_noise"] = False


@configclass
class NoisyProprioceptionDiscretizedTactileCfg(ObservationsCfg.NoisyProprioceptionCfg):
    def __post_init__(self):
        super().__post_init__()
        tactile_input_cfg = NoisyDiscretizedTactileCfg()
        self.tactile_input = tactile_input_cfg.tactile_signals

@configclass
class DenoisedProprioceptionDiscretizedTactileCfg(NoisyProprioceptionDiscretizedTactileCfg):
    def __post_init__(self):
        super().__post_init__()
        self.enable_corruption = False
        self.tactile_input.params["add_threshold_noise"] = False
        self.tactile_input.params["add_force_noise"] = False
        self.tactile_input.params["add_level_noise"] = False


# ----- Continuous-Tactile
@configclass
class NoisyContinuousTactileCfg(NoisyTactileCfg):
    def __post_init__(self):
        super().__post_init__()
        self.tactile_signals.func = mdp.CotinuousTactileSignals

@configclass
class NoisyProprioceptionContinuousTactileCfg(ObservationsCfg.NoisyProprioceptionCfg):
    def __post_init__(self):
        super().__post_init__()
        tactile_input_cfg = NoisyContinuousTactileCfg()
        self.tactile_input = tactile_input_cfg.tactile_signals

@configclass
class DenoisedProprioceptionContinuousTactileCfg(NoisyProprioceptionContinuousTactileCfg):
    def __post_init__(self):
        super().__post_init__()
        self.enable_corruption = False
        self.tactile_input.params["add_threshold_noise"] = False
        self.tactile_input.params["add_force_noise"] = False
        self.tactile_input.params["add_level_noise"] = False


# this class provide full formats, just for evaluation
@configclass
class NoisyProcessedTactileCfg(NoisyTactileCfg):
    def __post_init__(self):
        super().__post_init__()
        self.tactile_signals.func = mdp.ProcessedTactileSignals




# ------------------------ Rand Cylinder Transport Student------------------------
@configclass
class RandCylinderTransportStudentSingleBinaryTacEnvCfg(RandCylinderTransportTeacherEnvCfg):
    def __post_init__(self):
        if self.scene.num_envs==4096: self.scene.num_envs = 405
        super().__post_init__()
        self.observations.tactile = NoisyBinaryTactileCfg()
        self.observations.object_state = NoisyObjectStateCfg()
        teacher_env_from_train_to_play_for_distillation(self)


@configclass
class RandCylinderTransportStudentSingleBinaryTacEnvCfg_PLAY(RandCylinderTransportStudentSingleBinaryTacEnvCfg):
    def __post_init__(self):
        self.scene.num_envs = 20
        super().__post_init__()
        self.observations.original_tactile = NoisyTactileCfg()
        self.observations.processed_tactile = NoisyProcessedTactileCfg()


def teacher_env_from_train_to_play_for_distillation(env_cfg: ObjectTransportTeacherEnvCfg) -> None:
    # shorter episode length
    env_cfg.episode_length_s = 10.0

    # enable tactile sensors
    env_cfg.scene.robot = LocoTouch_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    create_tactile_contact_sensor(env_cfg)

    # use full range of commands and final setups
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

def create_tactile_contact_sensor(env_cfg: ObjectTransportTeacherEnvCfg) -> None:
    env_cfg.scene.tactile_contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/sensor_.*",
        update_period=0.025, # 40Hz
        history_length=3,
        track_air_time=True,
    )







