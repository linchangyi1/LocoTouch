import gymnasium as gym
import os
import cli_args
from datetime import datetime
import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
from omni.isaac.lab.envs import DirectMARLEnv, multi_agent_to_single_agent
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from locotouch.config.locotouch.agents.distillation_cfg import DistillationCfg
from locotouch.distill import *
from loco_rl.loco_rl.runners import OnPolicyRunner


class Distillation:
    def __init__(self, simulation_app, args_cli):
        self.simulation_app = simulation_app
        self.training = args_cli.training
        
        # distillation cfg
        distillation_cfg: DistillationCfg = cli_args.parse_distillation_cfg(args_cli.task, args_cli)
        distillation_log_root = os.path.join(distillation_cfg.log_root_path, distillation_cfg.experiment_name)
        distillation_log_root = os.path.abspath(distillation_log_root)

        # env
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
        self.env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
        # wrap for video recording under play mode
        if not self.training:
            self.record_video = args_cli.video
            if self.record_video:
                self.video_length = args_cli.video_length
                log_dir = os.path.join(distillation_log_root, distillation_cfg.log_dir_distill)
                video_kwargs = {
                    "video_folder": os.path.join(log_dir, "videos", "play"),
                    "step_trigger": lambda step: step == 0,
                    "video_length": args_cli.video_length,
                    "disable_logger": True,
                }
                print("[INFO] Recording videos during playing.")
                print_dict(video_kwargs, nesting=4)
                self.env = gym.wrappers.RecordVideo(self.env, **video_kwargs)
        self.env = multi_agent_to_single_agent(self.env) if isinstance(self.env.unwrapped, DirectMARLEnv) else self.env
        self.env = RslRlVecEnvWrapper(self.env)

        # dimensions
        _, extras = self.env.get_observations()
        observations = extras["observations"]
        proprioception_dim = observations["policy"].shape[-1] - observations["object_state"].shape[-1]
        tactile_signal_shape = observations["tactile"].shape[1:]
        tactile_signal_dim = tactile_signal_shape[0] if len(tactile_signal_shape) == 1 else tactile_signal_shape
        print(f"[INFO] Tactile signal dim: {tactile_signal_dim}, Proprioception dim: {proprioception_dim}")

        # delayed tactile recoder
        self.tactile_recorder = TactileRecorder(
            self.env.device, self.env.num_envs, tactile_signal_dim, distillation_cfg.min_delay, distillation_cfg.max_delay)

        # training mode
        if self.training:
            # create distillation log dir
            distillation_log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            distillation_cfg.log_dir = os.path.join(distillation_log_root, distillation_log_dir)
            if not os.path.exists(distillation_cfg.log_dir):
                os.makedirs(distillation_cfg.log_dir)
            print(f"[INFO] Logging student distillation in: {distillation_cfg.log_dir}")

            # logger
            self.use_wandb = distillation_cfg.logger == "wandb"
            self.use_tensorboard = distillation_cfg.logger == "tensorboard"
            self.logger = None
            if self.use_wandb:
                import wandb
                self.logger = wandb.init(project=distillation_cfg.wandb_project, config=distillation_cfg.to_dict(), name=distillation_cfg.log_dir.split("/")[-1])
                self.logger.config.update(distillation_cfg.to_dict())
            elif self.use_tensorboard:
                from torch.utils.tensorboard import SummaryWriter
                self.logger = SummaryWriter(log_dir=distillation_cfg.log_dir)

            # teacher policy
            teacher_policy_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
            rsl_log_root_path = os.path.join("logs", "rsl_rl", teacher_policy_cfg.experiment_name)
            rsl_log_root_path = os.path.abspath(rsl_log_root_path)
            resume_path = get_checkpoint_path(rsl_log_root_path, teacher_policy_cfg.load_run, teacher_policy_cfg.load_checkpoint)
            print(f"[INFO] Loading teacher policy checkpoint from: {resume_path}")
            ppo_runner = OnPolicyRunner(self.env, teacher_policy_cfg.to_dict(), log_dir=None, device=teacher_policy_cfg.device)
            ppo_runner.load(resume_path)
            MonolithicDistillation = distillation_cfg.distillation_type == "Monolithic"
            self.teacher_policy_inference = ppo_runner.get_inference_policy(device=self.env.device)  # needed for collecting data no matter Monolithic or RMA distillation
            self.teacher_encoder_inference = ppo_runner.get_inference_encoder(device=self.env.device) if not MonolithicDistillation else None
            self.teacher_backbone_weights = ppo_runner.get_backbone_weights() if not MonolithicDistillation else None

            # student policy and replay buffer
            self.student = Student(
                distillation_cfg,
                proprioception_dim,
                tactile_signal_dim,
                self.env.num_actions,
                teacher_policy_inference=self.teacher_policy_inference,
                teacher_encoder_inference=self.teacher_encoder_inference,
                teacher_backbone_weights=self.teacher_backbone_weights,
                logger=self.logger,
            )
            self.replay_buffer = ReplayBuffer(self.env, self.tactile_recorder, proprioception_dim)

            # dagger training parameters
            self.max_iterations = distillation_cfg.num_iterations
            self.bc_data_steps = distillation_cfg.bc_data_steps
            self.dagger_data_steps = distillation_cfg.dagger_data_steps
            self.evaluation_trajs_num = distillation_cfg.evaluation_trajs_num

        # play mode
        else:
            # student policy with loading checkpoint
            self.student = Student(
                distillation_cfg,
                proprioception_dim,
                tactile_signal_dim,
                self.env.num_actions,
            )
            resume_path = get_checkpoint_path(distillation_log_root, distillation_cfg.log_dir_distill, distillation_cfg.checkpoint_distill)
            self.student.load_checkpoint(resume_path)
            print(f"[INFO] Loading student policy checkpoint from: {resume_path}")

            # use ROS to publish the tactile signals
            self.publish_tactile_ros_topic = False
            if self.publish_tactile_ros_topic:
                import rospy
                from std_msgs.msg import Float32MultiArray
                rospy.init_node("tactile_publisher", anonymous=True)
                self.policy_tactile_pub = rospy.Publisher(distillation_cfg.policy_tactile_topic, Float32MultiArray, queue_size=1)
                self.policy_tactile_msg = Float32MultiArray()
                self.original_tactile_pub = rospy.Publisher(distillation_cfg.original_tactile_topic, Float32MultiArray, queue_size=1)
                self.original_tactile_msg = Float32MultiArray()
                self.processed_tactile_pub = rospy.Publisher(distillation_cfg.processed_tactile_topic, Float32MultiArray, queue_size=1)
                self.processed_tactile_msg = Float32MultiArray()

    def run(self):
        self.train() if self.training else self.play()
        self.env.close()

    def train(self):
        for iter in range(self.max_iterations):
            print("-" * 100)

            # collect data
            rewards, lengths = self.replay_buffer.collect_data(
                teacher_policy=self.teacher_policy_inference,
                student_policy=self.student if iter else None,
                num_steps=self.dagger_data_steps if iter else self.bc_data_steps)
            self.log_trajectory_rewards_and_lengths(rewards, lengths)

            # train student policy
            self.student.train_on_data(self.replay_buffer, iter)

            # evaluate student policy at the end
            if iter == self.max_iterations - 1:
                self.replay_buffer.clear_buffer()
                rewards, lengths = self.replay_buffer.evaluate(self.student, self.evaluation_trajs_num)
                self.log_trajectory_rewards_and_lengths(rewards, lengths)
                print("Log dir: ", self.student.log_dir)
        
        if self.use_wandb:
            self.logger.finish()
        elif self.use_tensorboard:
            self.logger.close()

    def log_trajectory_rewards_and_lengths(self, rewards: list, lengths: list):
        rewards = np.array(rewards)
        lengths = np.array(lengths)
        self.logger.log({"collect/trj_num": len(rewards)}, commit=False) if self.use_wandb else None
        self.logger.log({"collect/trj_less_half_len_num": (lengths < 0.5 * max(lengths)).sum()}, commit=False) if self.use_wandb else None
        self.logger.log({"collect/step_reward_mean": np.mean(rewards / lengths)}, commit=False) if self.use_wandb else None
        self.logger.log({"collect/trj_rwd_mean": rewards.mean(), "collect/trj_rwd_std": rewards.std()}, commit=False) if self.use_wandb else None
        self.logger.log({"collect/trj_len_mean": lengths.mean(), "collect/trj_len_std": lengths.std()}, commit=False) if self.use_wandb else None
        print(f"Collected {len(rewards)} trajectories:")
        print(f"Trajectories less than half length: {(lengths < 0.5 * max(lengths)).sum()}")
        print(f"Mean step reward: {np.mean(rewards / lengths)}")
        print(f"Mean reward: {rewards.mean()}, Std reward: {rewards.std()}")
        print(f"Mean length: {lengths.mean()}, Std length: {lengths.std()}")

    def play(self):
        self.student.eval()
        _, extras = self.env.get_observations()
        timestep = 0
        with torch.inference_mode():
            while self.simulation_app.is_running():
                # record tactile signals and apply delay
                self.tactile_recorder.record_new_tactile_signals(extras["observations"]["tactile"])
                extras["observations"]["tactile"] = self.tactile_recorder.get_tactile_signals().clone()

                # publish tactile signals
                if self.publish_tactile_ros_topic:
                    self.policy_tactile_msg.data = extras["observations"]["tactile"].cpu().numpy()[0].tolist()
                    self.policy_tactile_pub.publish(self.policy_tactile_msg)
                    if "original_tactile" in extras["observations"]:
                        self.original_tactile_msg.data = extras["observations"]["original_tactile"].cpu().numpy()[0].tolist()
                        self.original_tactile_pub.publish(self.original_tactile_msg)
                    if "processed_tactile" in extras["observations"]:
                        self.processed_tactile_msg.data = extras["observations"]["processed_tactile"].cpu().numpy()[0].tolist()
                        self.processed_tactile_pub.publish(self.processed_tactile_msg)
                
                # step env
                action = self.student.extract_input_and_forward(extras["observations"])
                obs, rwd, dones, extras = self.env.step(action)
                if dones.any():
                    # reset tactile recorder
                    done_idx = dones.nonzero(as_tuple=False).flatten()
                    self.tactile_recorder.reset(done_idx)

                if self.record_video:
                    timestep += 1
                    # Exit the play loop after recording one video
                    if timestep == self.video_length:
                        break

