from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def object_below_robot(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object")
    ) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    return obj.data.root_pos_w[:, 2] < robot.data.root_pos_w[:, 2]

def bad_roll(
    env: ManagerBasedRLEnv, limit_angle: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.asin(asset.data.projected_gravity_b[:, 1]).abs() > limit_angle

