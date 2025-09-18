from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class LocomotionPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 50
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    experiment_name = "locotouch_locomotion"
    logger = "wandb"
    # logger = "tensorboard"
    wandb_project = "Locomotion"


@configclass
class VelCurPPORunnerCfg(LocomotionPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "locotouch_vel_cur"

@configclass
class ObjectTransportTeacherPPORunnerCfg(LocomotionPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.resume = False
        self.resume_experiment = "locomotion_pretrain"
        self.experiment_name = "locotouch_object_transport_teacher"
        self.wandb_project = "Object_Transport"


@configclass
class CylinderTransportTeacherPPORunnerCfg(ObjectTransportTeacherPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "locotouch_cylinder_transport_teacher"


@configclass
class RandCylinderTransportTeacherPPORunnerCfg(ObjectTransportTeacherPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "locotouch_rand_cylinder_transport_teacher"









































# # ------------------------ Encoder ------------------------
# @configclass
# class CylinderTransportTeacherObjectStateEncoderPPORunnerCfg(CylinderTransportTeacherPPORunnerCfg):
#     def __post_init__(self):
#         super().__post_init__()
#         self.policy.class_name = "ActorCriticEncoder"
#         self.policy.actor_flatten_obs_end_idx = 270
#         self.policy.actor_encoder_obs_start_idx = 270
#         self.policy.actor_encoder_hidden_dims = [256, 128, 64]
#         self.policy.actor_encoder_embedding_dim = 64
#         self.policy.critic_flatten_obs_end_idx = None
#         self.policy.critic_encoder_obs_start_idx = None
#         self.policy.critic_encoder_hidden_dims = None
#         self.policy.critic_encoder_embedding_dim = None
#         self.policy.encoder_activation = "elu"
#         self.policy.encoder_final_activation = "tanh"

# @configclass
# class CylinderTransportTeacherSingleObjectStateEncoderPPORunnerCfg(CylinderTransportTeacherObjectStateEncoderPPORunnerCfg):
#     def __post_init__(self):
#         super().__post_init__()
#         self.policy.actor_flatten_obs_end_idx = -13*1
#         self.policy.actor_encoder_obs_start_idx = -13*1


# # ------------------------ RNNEncoder ------------------------
# @configclass
# class CylinderTransportTeacherObjectStateRNNEncoderPPORunnerCfg(CylinderTransportTeacherObjectStateEncoderPPORunnerCfg):
#     def __post_init__(self):
#         super().__post_init__()
#         self.policy.class_name = "ActorCriticRNNEncoder"
#         self.policy.critic_flatten_obs_end_idx = self.policy.actor_flatten_obs_end_idx
#         self.policy.critic_encoder_obs_start_idx = self.policy.actor_encoder_obs_start_idx
#         self.policy.critic_encoder_hidden_dims = self.policy.actor_encoder_hidden_dims
#         self.policy.critic_encoder_embedding_dim = self.policy.actor_encoder_embedding_dim
#         self.policy.encoder_rnn_type="gru"
#         self.policy.encoder_rnn_hidden_size=256
#         self.policy.encoder_rnn_num_layers=1






# # ------------------------ Pretrained Transport ------------------------
# # still need to use --resume because the default args_cli.resume is False if not given
# @configclass
# class CylinderTransportTeacherFCPreTrainedPPORunnerCfg(CylinderTransportTeacherFCPPORunnerCfg):
#     def __post_init__(self):
#         super().__post_init__()
#         self.resume = True
#         self.resume_experiment = "locomotion_pretrain"
#         self.pretrained = True



# # ------------------------ Tactile Input ------------------------


# @configclass
# class CylinderTransportTactileInputPPORunnerCfg(CylinderTransportTeacherObjectStateEncoderPPORunnerCfg):
#     def __post_init__(self):
#         super().__post_init__()
#         self.experiment_name = "locotouch_object_transport_teacher_tactile"
#         self.policy.class_name = "ActorCriticPreEncoderRNNEncoder"
#         self.policy.actor_flatten_obs_end_idx = -17*13
#         self.policy.actor_encoder_obs_start_idx = -17*13
#         self.policy.actor_pre_encoder_hidden_dims = [128, 128, 64]
#         self.policy.actor_pre_encoder_embedding_dim = 64
#         self.policy.critic_flatten_obs_end_idx = self.policy.actor_flatten_obs_end_idx
#         self.policy.critic_encoder_obs_start_idx = self.policy.actor_encoder_obs_start_idx
#         self.policy.critic_pre_encoder_hidden_dims = self.policy.actor_pre_encoder_hidden_dims
#         self.policy.critic_pre_encoder_embedding_dim = self.policy.actor_pre_encoder_embedding_dim
#         self.policy.critic_encoder_hidden_dims = self.policy.actor_encoder_hidden_dims
#         self.policy.critic_encoder_embedding_dim = self.policy.actor_encoder_embedding_dim
#         self.policy.pre_encoder_activation="elu"
#         self.policy.encoder_rnn_type="gru"
#         self.policy.encoder_rnn_hidden_size=256
#         self.policy.encoder_rnn_num_layers=1


# @configclass
# class CylinderTransportTactileInputCNNPPORunnerCfg(CylinderTransportTeacherObjectStateEncoderPPORunnerCfg):
#     def __post_init__(self):
#         super().__post_init__()
#         self.experiment_name = "locotouch_object_transport_teacher_tactile"
#         self.policy.class_name = "ActorCriticPreEncoderRNNEncoder"
#         self.policy.actor_flatten_obs_end_idx = -2*17*13
#         self.policy.actor_encoder_obs_start_idx = -2*17*13
#         self.policy.actor_pre_encoder_hidden_dims = [128, 128, 64]
#         self.policy.actor_pre_encoder_embedding_dim = 64
#         self.policy.critic_flatten_obs_end_idx = self.policy.actor_flatten_obs_end_idx
#         self.policy.critic_encoder_obs_start_idx = self.policy.actor_encoder_obs_start_idx
#         self.policy.critic_pre_encoder_hidden_dims = self.policy.actor_pre_encoder_hidden_dims
#         self.policy.critic_pre_encoder_embedding_dim = self.policy.actor_pre_encoder_embedding_dim
#         self.policy.critic_encoder_hidden_dims = self.policy.actor_encoder_hidden_dims
#         self.policy.critic_encoder_embedding_dim = self.policy.actor_encoder_embedding_dim
#         self.policy.pre_encoder_activation="elu"
#         self.policy.encoder_rnn_type="gru"
#         self.policy.encoder_rnn_hidden_size=256
#         self.policy.encoder_rnn_num_layers=1



# @configclass
# class CylinderTransportTactileInputAsymmetricPPORunnerCfg(CylinderTransportTactileInputPPORunnerCfg):
#     def __post_init__(self):
#         super().__post_init__()
#         self.experiment_name = "locotouch_object_transport_teacher_tactile"
#         self.policy.class_name = "ActorCriticPreEncoderRNNEncoder"
#         self.policy.critic_flatten_obs_end_idx = None
#         self.policy.critic_encoder_obs_start_idx = None
#         self.policy.critic_pre_encoder_hidden_dims = None
#         self.policy.critic_pre_encoder_embedding_dim = None
#         self.policy.critic_encoder_hidden_dims = None
#         self.policy.critic_encoder_embedding_dim = None


# @configclass
# class CylinderTransportTactileDoubleChannelsPPORunnerCfg(CylinderTransportTactileInputPPORunnerCfg):
#     def __post_init__(self):
#         super().__post_init__()
#         self.policy.actor_flatten_obs_end_idx = -17*13*2
#         self.policy.actor_encoder_obs_start_idx = -17*13*2
#         self.policy.critic_flatten_obs_end_idx = self.policy.actor_flatten_obs_end_idx
#         self.policy.critic_encoder_obs_start_idx = self.policy.actor_encoder_obs_start_idx

# @configclass
# class CylinderTransportTactileDoubleChannelsAsymmetricPPORunnerCfg(CylinderTransportTactileInputAsymmetricPPORunnerCfg):
#     def __post_init__(self):
#         super().__post_init__()
#         self.policy.actor_flatten_obs_end_idx = -17*13*2
#         self.policy.actor_encoder_obs_start_idx = -17*13*2


# # ------------------------ Pretrained Locomotion ------------------------

# @configclass
# class ObjectTransportLocomotionPretrainPPORunnerCfg(LocomotionPPORunnerCfg):
#     def __post_init__(self):
#         super().__post_init__()
#         self.experiment_name = "locomotion_pretrain"
#         self.logger = "tensorboard"





# # ------------------------ Other Objects ------------------------


# @configclass
# class CuboidTransportTeacherPPORunnerCfg(ObjectTransportTeacherPPORunnerCfg):
#     def __post_init__(self):
#         super().__post_init__()
#         self.experiment_name = "locotouch_cuboid_transport_teacher"

# @configclass
# class SphereTransportTeacherPPORunnerCfg(ObjectTransportTeacherPPORunnerCfg):
#     def __post_init__(self):
#         super().__post_init__()
#         self.experiment_name = "locotouch_sphere_transport_teacher"

# @configclass
# class RandCuboidTransportTeacherPPORunnerCfg(ObjectTransportTeacherPPORunnerCfg):
#     def __post_init__(self):
#         super().__post_init__()
#         self.experiment_name = "locotouch_rand_cuboid_transport_teacher"



# @configclass
# class RandSphereTransportTeacherPPORunnerCfg(ObjectTransportTeacherPPORunnerCfg):
#     def __post_init__(self):
#         super().__post_init__()
#         self.experiment_name = "locotouch_rand_sphere_transport_teacher"





