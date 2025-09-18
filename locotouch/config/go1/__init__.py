import gymnasium as gym
from .agents import rsl_rl_ppo_cfg
from . import locomotion_env_cfg, locomotion_vel_cur_env_cfg


# ----------------------------------- Locomotion -----------------------------------
gym.register(
    id="Isaac-Locomotion-Go1-v1",  # environment name
    entry_point="isaaclab.envs:ManagerBasedRLEnv",  # env type: <module>:<class>
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": locomotion_env_cfg.LocomotionEnvCfg,  # environment configuration
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.LocomotionPPORunnerCfg,  # RL configuration
    },
)

gym.register(
    id="Isaac-Locomotion-Go1-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": locomotion_env_cfg.LocomotionEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.LocomotionPPORunnerCfg,
    },
)
"""
python locotouch/scripts/train.py --task Isaac-Locomotion-Go1-v1 --num_envs=20 --logger=tensorboard
python locotouch/scripts/train.py --task Isaac-Locomotion-Go1-v1 --num_envs=4096 --headless

python locotouch/scripts/play.py --task Isaac-Locomotion-Go1-Play-v1 --num_envs=20 --load_run=2025-02-09_21-11-23
"""


# ----------------------------------- Locomotion with Velocity Curriculum -----------------------------------
gym.register(
    id="Isaac-LocomotionVelCur-Go1-v1",  # environment name
    entry_point="isaaclab.envs:ManagerBasedRLEnv",  # env type: <module>:<class>
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": locomotion_vel_cur_env_cfg.LocomotionVelCurEnvCfg,  # environment configuration
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.VelCurPPORunnerCfg,  # RL configuration
    },
)

gym.register(
    id="Isaac-LocomotionVelCur-Go1-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": locomotion_vel_cur_env_cfg.LocomotionVelCurEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.VelCurPPORunnerCfg,
    },
)
"""
python locotouch/scripts/train.py --task Isaac-LocomotionVelCur-Go1-v1 --num_envs=20 --logger=tensorboard
python locotouch/scripts/train.py --task Isaac-LocomotionVelCur-Go1-v1 --num_envs=4096 --headless

python locotouch/scripts/play.py --task Isaac-LocomotionVelCur-Go1-Play-v1 --num_envs=20
"""
