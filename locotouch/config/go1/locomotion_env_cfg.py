from omni.isaac.lab.utils import configclass
from locotouch.config.base.locomotion_base_env_cfg import LocomotionBaseEnvCfg, smaller_scene_for_playing
from locotouch.assets.locotouch import LocoTouch_Without_Tactile_Instanceable_CFG as Train_Robot_CFG
from locotouch.assets.locotouch import LocoTouch_Without_Tactile_CFG as Play_Robot_CFG


def go1_locomotion_train_env_post_init_func(env_cfg: LocomotionBaseEnvCfg) -> None:
    env_cfg.scene.robot = Train_Robot_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

def go1_locomotion_play_env_post_init_func(env_cfg: LocomotionBaseEnvCfg) -> None:
    # change the uninstancable usd for colorful visualization
    env_cfg.scene.robot = Play_Robot_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    smaller_scene_for_playing(env_cfg)


@configclass
class LocomotionEnvCfg(LocomotionBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        go1_locomotion_train_env_post_init_func(self)


@configclass
class LocomotionEnvCfg_PLAY(LocomotionEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        go1_locomotion_play_env_post_init_func(self)



