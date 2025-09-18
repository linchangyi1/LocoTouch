from isaaclab.utils import configclass
from locotouch.config.base.locomotion_vel_cur_base_env_cfg import LocomotionVelCurBaseEnvCfg, locomotion_vel_cur_play_env_post_init_func
from .locomotion_env_cfg import locotouch_locomotion_train_env_post_init_func, locotouch_locomotion_play_env_post_init_func


def locotouch_locomotion_vel_cur_train_env_post_init_func(env_cfg: LocomotionVelCurBaseEnvCfg) -> None:
    locotouch_locomotion_train_env_post_init_func(env_cfg)

def locotouch_locomotion_vel_cur_play_env_post_init_func(env_cfg: LocomotionVelCurBaseEnvCfg) -> None:
    locotouch_locomotion_play_env_post_init_func(env_cfg)
    locomotion_vel_cur_play_env_post_init_func(env_cfg)


@configclass
class LocomotionVelCurEnvCfg(LocomotionVelCurBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        locotouch_locomotion_vel_cur_train_env_post_init_func(self)


class LocomotionVelCurEnvCfg_PLAY(LocomotionVelCurEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        locotouch_locomotion_vel_cur_play_env_post_init_func(self)

