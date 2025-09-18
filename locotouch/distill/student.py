import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
from loco_rl.models.model_generation import generate_model
from loco_rl.models import MLP, RNN, CNN2d, CNN2dHead
from locotouch.config.locotouch.agents.distillation_cfg import DistillationCfg
from locotouch.distill.replay_buffer import ReplayBuffer


class Student(nn.Module):
    def __init__(
        self,
        cfg: DistillationCfg,
        proprioception_dim: int,
        tactile_signal_dim: int,
        action_dim: int,
        teacher_policy_inference=None,  # for training
        teacher_encoder_inference=None,  # for RMA distillation
        teacher_backbone_weights=None,  # for RMA distillation
        logger=None):
        super().__init__()
        # parameters
        self.cfg = cfg
        self.logger = logger
        self.device = self.cfg.device
        self.log_dir = self.cfg.log_dir
        self.proprioception_dim = proprioception_dim
        self.tactile_signal_dim = tactile_signal_dim
        self.tactile_signal_img_shape = cfg.pre_encoder.img_shape
        self.tactile_embedding_dim = cfg.tactile_encoder.embedding_dim
        self.action_dim = action_dim

        print("-------------- Construct Student Network --------------")
        # pre encoder
        pre_encoder_type = cfg.pre_encoder.model_type
        self.use_pre_encoder = True if "CNN" in pre_encoder_type else (cfg.pre_encoder.hidden_dims is not None)
        if self.use_pre_encoder:
            self.pre_encoder: MLP|RNN|CNN2d|CNN2dHead = generate_model(
                self.tactile_signal_dim,
                cfg.pre_encoder.embedding_dim,
                cfg.pre_encoder).to(self.device)
            print(f"Pre Encoder: {self.pre_encoder}")

        # student encoder
        self.student_encoder: MLP|RNN|CNN2d = generate_model(
            self.tactile_signal_dim if not self.use_pre_encoder else cfg.pre_encoder.embedding_dim,
            self.tactile_embedding_dim,
            cfg.tactile_encoder).to(self.device)
        print(f"Student Encoder: {self.student_encoder}")

        # student backbone
        self.student_backbone: MLP|RNN = generate_model(
            self.proprioception_dim + self.tactile_embedding_dim,
            self.action_dim,
            cfg.student_policy).to(self.device)
        print(f"Student Backbone: {self.student_backbone}")

        self.MonolithicDistillation = cfg.distillation_type == "Monolithic"
        self.RMA_distillation = not self.MonolithicDistillation # for more intuitive hint
        self.teacher_policy_inference = teacher_policy_inference
        self.teacher_encoder_inference = teacher_encoder_inference
        self.teacher_backbone_weights = teacher_backbone_weights
        # load teacher backbone weights to student backbone and freeze it
        if self.teacher_backbone_weights is not None:
            self.student_backbone.model.load_state_dict(self.teacher_backbone_weights)
            for param in self.student_backbone.parameters():
                param.requires_grad = False

        # training parameters within each iteration
        self.max_iterations = self.cfg.num_iterations
        self.initial_epoches = self.cfg.initial_epoches
        self.incremental_epoches = self.cfg.incremental_epoches
        self.final_epoches = self.cfg.final_epoches
        self.batch_steps = self.cfg.batch_steps

        # optimizer and criterion
        self._criterion = nn.MSELoss(reduction='none')
        self._distill_lr  = cfg.distill_lr
        self._optimizer = torch.optim.AdamW(self.parameters(), lr=self._distill_lr)

        # actions parameters
        self.clip_actions = cfg.clip_actions
        self.clip_range = cfg.clip_range
        self.action_scale_within_env = cfg.action_scale_within_env

    def encoder_forward(self, tactile_signal, hidden_states=None):
        if self.use_pre_encoder:
            original_tactile_shape = tactile_signal.shape  # NxCxHxW or LxBxCxHxW
            if len(tactile_signal.shape) <=3:
                flatten_tactile_shape = tactile_signal.shape  # NxD or LxBxD
                tactile_signal = tactile_signal.reshape(*flatten_tactile_shape[:-1], *self.tactile_signal_img_shape)  # NxCxHxW or LxBxCxHxW
                original_tactile_shape = tactile_signal.shape  # NxCxHxW or LxBxCxHxW
            cnn_tactile_signals = tactile_signal.reshape(-1, *original_tactile_shape[-3:])  # NxCxHxW
            tactile_signal = self.pre_encoder(cnn_tactile_signals).reshape(*original_tactile_shape[:-3], -1)  # NxD
        return self.student_encoder(tactile_signal, hidden_states)
    
    def backbone_forward(self, proprioception, tactile_embedding):
        policy_input = torch.cat((proprioception, tactile_embedding), dim=-1)
        return self.student_backbone(policy_input)

    def forward(self, proprioception, tactile_signal, hidden_states=None):
        tactile_embedding = self.encoder_forward(tactile_signal, hidden_states)
        policy_input = torch.cat((proprioception, tactile_embedding), dim=-1)
        return self.student_backbone(policy_input)

    def train_on_data(self, replay_buffer: ReplayBuffer, num_iter: int):
        self.train()
        num_epoches = self.initial_epoches + self.incremental_epoches * num_iter
        num_epoches += self.final_epoches if num_iter == self.max_iterations - 1 else 0
        pbar = tqdm(range(num_epoches), desc="Training")
        batch_trajs = int(self.batch_steps / (replay_buffer.num_steps / replay_buffer.num_trajs)) + 1
        for _ in pbar:
            # train
            losses = []
            action_mses = []
            action_maes = []
            dataloader = replay_buffer.to_recurrent_generator(batch_size=batch_trajs)
            for batch in dataloader:
                self._optimizer.zero_grad()
                proprioceptions = batch['proprioceptions']
                teacher_encoder_obses = batch['teacher_encoder_obses']
                tactile_signals = batch['tactile_signals']
                masks = batch['masks']
                if self.MonolithicDistillation:
                    student_actions = self.forward(proprioceptions, tactile_signals)
                    teacher_obses = torch.cat((proprioceptions, teacher_encoder_obses), dim=-1)
                    teacher_actions = self.teacher_policy_inference(teacher_obses)
                    loss = self._criterion(student_actions, teacher_actions).mean(dim=-1)
                else:
                    student_embeddings = self.encoder_forward(tactile_signals)
                    teacher_embeddings = self.teacher_encoder_inference(teacher_encoder_obses) if self.teacher_encoder_inference is not None else teacher_encoder_obses
                    loss = self._criterion(student_embeddings, teacher_embeddings).mean(dim=-1)
                    student_actions = self.backbone_forward(proprioceptions, student_embeddings)
                    teacher_obses = torch.cat((proprioceptions, teacher_encoder_obses), dim=-1)
                    teacher_actions = self.teacher_policy_inference(teacher_obses)
                    action_mse = ((student_actions - teacher_actions) ** 2).mean(dim=-1)
                    action_mse = (action_mse * masks).sum() / masks.sum()
                    action_mses.append(action_mse.item())
                loss = (loss * masks).sum() / masks.sum()
                losses.append(loss.item())
                loss.backward()
                self._optimizer.step()
                # compute action MAE
                if self.clip_actions:
                    student_actions = torch.clamp(student_actions, -self.clip_range, self.clip_range)
                    teacher_actions = torch.clamp(teacher_actions, -self.clip_range, self.clip_range)
                action_mae = torch.abs(student_actions - teacher_actions).mean(dim=-1)
                action_mae = (action_mae * masks).sum() / masks.sum() * self.action_scale_within_env
                action_maes.append(action_mae.item())
            # logging
            if self.logger is not None:
                if self.MonolithicDistillation:
                    self.logger.log({"train/Action MSE": np.mean(losses),
                                    "train/Action MAE": np.mean(action_maes)}) if self.cfg.logger == "wandb" else None
                else:
                    self.logger.log({"train/Action MSE": np.mean(action_mses),
                                    "train/Action MAE": np.mean(action_maes),
                                    "train/Encoder MSE": np.mean(losses)}) if self.cfg.logger == "wandb" else None
            pbar.set_postfix({f"Avg Loss": f"{np.mean(losses):.4f}"})
        pbar.close()
        print(f"[Distillation iteration {num_iter}] Action MSE: {np.mean(losses if self.MonolithicDistillation else action_mses)}")
        print(f"[Distillation iteration {num_iter}] Action MAE: {np.mean(action_maes)}")
        if not self.MonolithicDistillation: print(f"[Distillation iteration {num_iter}] Encoder MSE: {np.mean(losses)}")
        self.save_model(num_iter)

    def save_model(self, iteration):
        model_path = os.path.join(self.log_dir, f"model_{iteration}.pt")
        torch.save(self.state_dict(), model_path)

    def load_checkpoint(self, model_path):
        self.load_state_dict(torch.load(model_path, map_location=self.device))

    def extract_input_and_forward(self, obs):
        tactile_signal = obs["tactile"]
        proprioception = obs["policy"][:, :self.proprioception_dim]
        return self.forward(proprioception, tactile_signal)

    def reset(self, dones=None):
        if self.use_pre_encoder: self.pre_encoder.reset(dones)
        self.student_encoder.reset(dones)
        self.student_backbone.reset(dones)

    def get_hidden_states(self):
        if hasattr(self.student_encoder, "get_hidden_states"):
            return self.student_encoder.get_hidden_states()
