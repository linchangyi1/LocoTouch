import gymnasium as gym
from .agents import rsl_rl_ppo_cfg, distillation_cfg
from . import locomotion_env_cfg, locomotion_vel_cur_env_cfg, cylinder_transport_teacher_env_cfg, rand_cylinder_transport_teacher_env_cfg, object_transport_student_env_cfg


# ----------------------------------- Locomotion -----------------------------------
gym.register(
    id="Isaac-Locomotion-LocoTouch-v1",  # environment name
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",  # env type: <module>:<class>
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": locomotion_env_cfg.LocomotionEnvCfg,  # environment configuration
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.LocomotionPPORunnerCfg,  # RL configuration
    },
)

gym.register(
    id="Isaac-Locomotion-LocoTouch-Play-v1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": locomotion_env_cfg.LocomotionEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.LocomotionPPORunnerCfg,
    },
)
"""
python locotouch/scripts/train.py --task Isaac-Locomotion-LocoTouch-v1 --num_envs=20 --logger=tensorboard
python locotouch/scripts/train.py --task Isaac-Locomotion-LocoTouch-v1 --num_envs=4096 --headless

python locotouch/scripts/play.py --task Isaac-Locomotion-LocoTouch-Play-v1 --num_envs=20 --load_run=2025-02-09_21-11-23
"""


# ----------------------------------- Locomotion with Velocity Curriculum -----------------------------------
gym.register(
    id="Isaac-LocomotionVelCur-LocoTouch-v1",  # environment name
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",  # env type: <module>:<class>
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": locomotion_vel_cur_env_cfg.LocomotionVelCurEnvCfg,  # environment configuration
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.VelCurPPORunnerCfg,  # RL configuration
    },
)

gym.register(
    id="Isaac-LocomotionVelCur-LocoTouch-Play-v1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": locomotion_vel_cur_env_cfg.LocomotionVelCurEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.VelCurPPORunnerCfg,
    },
)
"""
python locotouch/scripts/train.py --task Isaac-LocomotionVelCur-LocoTouch-v1 --num_envs=20 --logger=tensorboard
python locotouch/scripts/train.py --task Isaac-LocomotionVelCur-LocoTouch-v1 --num_envs=4096 --headless

python locotouch/scripts/play.py --task Isaac-LocomotionVelCur-LocoTouch-Play-v1 --num_envs=20
"""


# ----------------------------------- Cylinder Transport Teacher-----------------------------------

gym.register(
    id="Isaac-CylinderTransportTeacher-LocoTouch-v1",  # environment name
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",  # env type: <module>:<class>
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": cylinder_transport_teacher_env_cfg.CylinderTransportTeacherEnvCfg,  # environment configuration
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.CylinderTransportTeacherPPORunnerCfg,  # RL configuration
    },
)

gym.register(
    id="Isaac-CylinderTransportTeacher-LocoTouch-Play-v1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": cylinder_transport_teacher_env_cfg.CylinderTransportTeacherEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.CylinderTransportTeacherPPORunnerCfg,
    },
)
"""
python locotouch/scripts/train.py --task Isaac-CylinderTransportTeacher-LocoTouch-v1 --num_envs=20 --logger=tensorboard
python locotouch/scripts/train.py --task Isaac-CylinderTransportTeacher-LocoTouch-v1 --num_envs=4096 --headless

python locotouch/scripts/play.py --task Isaac-CylinderTransportTeacher-LocoTouch-Play-v1 --num_envs=20
"""


# ----------------------------------- Random Cylinder Transport Teacher -----------------------------------
gym.register(
    id="Isaac-RandCylinderTransportTeacher-LocoTouch-v1",  # environment name
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",  # env type: <module>:<class>
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rand_cylinder_transport_teacher_env_cfg.RandCylinderTransportTeacherEnvCfg,  # environment configuration
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.RandCylinderTransportTeacherPPORunnerCfg,  # RL configuration
    },
)

gym.register(
    id="Isaac-RandCylinderTransportTeacher-LocoTouch-Play-v1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rand_cylinder_transport_teacher_env_cfg.RandCylinderTransportTeacherEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.RandCylinderTransportTeacherPPORunnerCfg,
    },
)
"""
python locotouch/scripts/train.py --task Isaac-RandCylinderTransportTeacher-LocoTouch-v1 --num_envs=20 --logger=tensorboard
python locotouch/scripts/train.py --task Isaac-RandCylinderTransportTeacher-LocoTouch-v1 --num_envs=4096 --headless

python locotouch/scripts/play.py --task Isaac-RandCylinderTransportTeacher-LocoTouch-Play-v1 --num_envs=20
"""



# ----------------------------------- Random Cylinder Transport Student -----------------------------------
gym.register(
    id="Isaac-RandCylinderTransportStudent_SingleBinaryTac_CNNRNN_Mon-LocoTouch-v1",  # environment name
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",  # env type: <module>:<class>
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": object_transport_student_env_cfg.RandCylinderTransportStudentSingleBinaryTacEnvCfg,  # environment configuration
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.RandCylinderTransportTeacherPPORunnerCfg,  # RL configuration
        "distillation_cfg_entry_point": distillation_cfg.DistillationRandCylinderCNNRNNMonCfg,  # distill configuration
    },
)
gym.register(
    id="Isaac-RandCylinderTransportStudent_SingleBinaryTac_CNNRNN_Mon-LocoTouch-Play-v1",  # environment name
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",  # env type: <module>:<class>
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": object_transport_student_env_cfg.RandCylinderTransportStudentSingleBinaryTacEnvCfg_PLAY,  # environment configuration
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.RandCylinderTransportTeacherPPORunnerCfg,  # RL configuration
        "distillation_cfg_entry_point": distillation_cfg.DistillationRandCylinderCNNRNNMonCfg,  # distill configuration
    },
)

"""
python locotouch/scripts/train.py --task Isaac-RandCylinderTransportStudent_SingleBinaryTac_CNNRNN_Mon-LocoTouch-v1 --num_envs=20 --logger=tensorboard
python locotouch/scripts/train.py --task Isaac-RandCylinderTransportStudent_SingleBinaryTac_CNNRNN_Mon-LocoTouch-v1 --num_envs=4096 --headless

python locotouch/scripts/play.py --task Isaac-RandCylinderTransportStudent_SingleBinaryTac_CNNRNN_Mon-LocoTouch-Play-v1 --num_envs=20
"""


