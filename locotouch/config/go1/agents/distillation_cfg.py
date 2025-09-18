from isaaclab.utils import configclass
from loco_rl.models import ModelCfg


@configclass
class PreEncoderCfg(ModelCfg):
    def __post_init__(self):
        super().__post_init__()
        self.model_type = "MLP"
        self.hidden_dims = None
        self.embedding_dim = None


@configclass
class TactileEncoderCfg(ModelCfg):
    def __post_init__(self):
        super().__post_init__()
        self.model_type = "MLP"
        self.hidden_dims = [256, 128, 64]
        self.embedding_dim = 64


@configclass
class StudentPolicyCfg(ModelCfg):
    def __post_init__(self):
        super().__post_init__()


@configclass
class DistillationCfg:
    # basic configurations
    distillation_type: str = "Monolithic"  # "Monolithic" or "RMA"
    pre_encoder: PreEncoderCfg = PreEncoderCfg()
    tactile_encoder: TactileEncoderCfg = TactileEncoderCfg()
    student_policy: StudentPolicyCfg = StudentPolicyCfg()

    # device and logging
    device: str = "cuda:0"
    log_root_path: str = "logs/distillation"
    experiment_name: str = "object"
    log_dir: str = "specify_log_dir"
    log_dir_distill: str = "specify_log_dir_distill"
    checkpoint_distill: str = "specify_checkpoint_distill"
    logger = "wandb"
    wandb_project = "Transport_Distillation"
    # logger = "tensorboard"

    # rollout collection
    num_iterations: int = 8
    bc_data_steps: int = 400000
    dagger_data_steps: int = 200000
    initial_epoches: int = 2000
    incremental_epoches: int = 500
    final_epoches: int = 0
    batch_steps: int = 20000
    distill_lr = 5.0e-4
    # final_lr = 5.0e-4
    # fix_lr_steps = 30000
    evaluation_trajs_num: int = 2000

    # actions
    clip_actions: bool = False
    # clip_actions: bool = True
    clip_range: float = 100.0
    action_scale_within_env: float = 0.25

    # tactile signal
    min_delay: int = 1
    max_delay: int = 2

    # ros topics for visualization
    policy_tactile_topic: str = "/policy_tactile_signal"
    original_tactile_topic: str = "/original_tactile_signal"
    processed_tactile_topic: str = "/processed_tactile_signal"


@configclass
class DistillationRandCylinderCNNRNNMonCfg(DistillationCfg):
    def __post_init__(self):
        super().__post_init__()
        self.pre_encoder.model_type = "CNN2dHead"
        self.pre_encoder.embedding_dim = 64
        self.tactile_encoder.model_type = "RNN"
        self.tactile_encoder.rnn_hidden_size = 512
        self.experiment_name = "rand_cylinder"





