from __future__ import annotations
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv
from omni.isaac.lab.managers.action_manager import ActionTerm
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.envs.mdp.actions import JointPositionActionCfg
from omni.isaac.lab.envs.mdp.actions import JointPositionAction


class JointPositionActionPrevPrev(JointPositionAction):
    def __init__(self, cfg: JointPositionActionPrevPrevCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        self._clip_raw_actions = cfg.clip_raw_actions
        if self._clip_raw_actions:
            self._raw_action_clip_value = cfg.raw_action_clip_value
        self._raw_action_scale = cfg.raw_action_scale

        self._prev_raw_actions = torch.zeros_like(self._raw_actions)
        self._prev_prev_raw_actions = torch.zeros_like(self._raw_actions)

        self._processed_actions = self._raw_actions * self._scale + self._offset
        self._prev_processed_actions = self._processed_actions.clone()
        self._prev_prev_processed_actions = self._processed_actions.clone()

    def process_actions(self, actions: torch.Tensor):
        # store the previous actions
        self._prev_prev_raw_actions[:] = self._prev_raw_actions.clone()
        self._prev_raw_actions[:] = self._raw_actions.clone()

        self._prev_prev_processed_actions[:] = self._prev_processed_actions.clone()
        self._prev_processed_actions[:] = self._processed_actions.clone()

        # clip the raw actions
        if self._clip_raw_actions:
            actions = torch.clamp(actions, -self._raw_action_clip_value, self._raw_action_clip_value)
        actions = actions * self._raw_action_scale

        # process the actions
        super().process_actions(actions)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._prev_raw_actions[env_ids] = 0.0
        self._prev_prev_raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = (self._raw_actions * self._scale + self._offset)[env_ids]
        self._prev_processed_actions[env_ids] = self._processed_actions[env_ids].clone()
        self._prev_prev_processed_actions[env_ids] = self._processed_actions[env_ids].clone()
        super().reset(env_ids)

    @property
    def prev_raw_actions(self) -> torch.Tensor:
        return self._prev_raw_actions
    
    @property
    def prev_prev_raw_actions(self) -> torch.Tensor:
        return self._prev_prev_raw_actions
    
    @property
    def prev_processed_actions(self) -> torch.Tensor:
        return self._prev_processed_actions
    
    @property
    def prev_prev_processed_actions(self) -> torch.Tensor:
        return self._prev_prev_processed_actions


@configclass
class JointPositionActionPrevPrevCfg(JointPositionActionCfg):
    class_type: type[ActionTerm] = JointPositionActionPrevPrev
    clip_raw_actions: bool = False
    raw_action_clip_value: float = 100.0  # only used if clip_raw_actions is True
    raw_action_scale : float = 1.0  # scale the actions before storing them as raw actions

