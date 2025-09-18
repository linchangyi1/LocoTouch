"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher
import cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
torch.set_printoptions(sci_mode=False, precision=5)

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

# Import extensions to set up environment tasks
from locotouch import *  # noqa: F401


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # robot = env.unwrapped.scene["robot"]
    # body_name = robot.body_names
    # joint_name = robot.joint_names
    # print(f"[INFO] Playing with the policy on the robot: {body_name}")
    # print(f"[INFO] Playing with the policy on the joint: {joint_name}")

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    actor_critic_class = agent_cfg.policy.class_name
    # if actor_critic_class == "ActorCriticEncoder":
    #     from loco_rl.runners import OnPolicyRunner as LocoOnPolicyRunner
    #     ppo_runner = LocoOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # else:
    #     from rsl_rl.runners import OnPolicyRunner as RslOnPolicyRunner
    #     ppo_runner = RslOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    from loco_rl.runners import OnPolicyRunner
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)


    # load previously trained model
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    # export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    # export_policy_as_jit(
    #     ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    # )
    # export_policy_as_onnx(
    #     ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    # )

    # reset environment
    # obs, extras = env.get_observations()
    env_obs = env.get_observations()
    obs = env_obs["policy"]

    # print("Shape of the observation: ", obs[0].shape)
    timestep = 0
    # simulate environment
    print_obs = False
    # print_obs = True
    mask_obs = False
    # mask_obs = True
    dones = None
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # term_dims = [3, 3, 3, 12, 12, 12, 3, 3, 4, 3]
            term_dims = [3, 3, 3, 12, 12, 12, 13]
            term_histories = [6] * len(term_dims)
            term_begin_idx = [sum(term_dim*term_histories for term_dim, term_histories in zip(term_dims[:i], term_histories[:i]) ) for i in range(len(term_dims))]
            term_end_idx = [sum(term_dim*term_histories for term_dim, term_histories in zip(term_dims[:i+1], term_histories[:i+1]) ) for i in range(len(term_dims))]
            # print("*" * 80)
            # print("term_begin_idx: ", term_begin_idx)
            # print("term_end_idx: ", term_end_idx)

            obj_term_dims = [3, 3, 4, 3]
            obj_state_dim = sum(obj_term_dims)
            obj_state_his_len = term_histories[-1]
            obj_term_begin_idx = [sum(term_dim for term_dim in obj_term_dims[:i]) for i in range(len(obj_term_dims))]
            obj_term_end_idx = [sum(term_dim for term_dim in obj_term_dims[:i+1]) for i in range(len(obj_term_dims))]
            # print("*" * 80)
            # print("obj_term_begin_idx: ", obj_term_begin_idx)
            # print("obj_term_end_idx: ", obj_term_end_idx)

            # get_obj_term_idx = lambda i: sum(term_dim*term_histories for term_dim, term_histories in zip(term_dims[:i], term_histories[:i])) + 6*6
            if print_obs:
                # print("Observation: ", obs[0])
                # print("*" * 80)
                # print("velocity_commands: \n", obs[0][term_begin_idx[0]:term_end_idx[0]])
                # print("base_ang_vel: \n", obs[0][term_begin_idx[1]:term_end_idx[1]])
                # print("projected_gravity: \n", obs[0][term_begin_idx[2]:term_end_idx[2]])
                # print("joint_pos: \n", obs[0][term_begin_idx[3]:term_end_idx[3]])
                # print("joint_vel: \n", obs[0][term_begin_idx[4]:term_end_idx[4]])
                # print("last_action: \n", obs[0][term_begin_idx[5]:term_end_idx[5]])
                # print("last_actiono_max_value:", torch.max(torch.abs(obs[:, term_begin_idx[5]:term_end_idx[5]])))
                if torch.max(torch.abs(obs[:, term_begin_idx[5]:term_end_idx[5]])) > 1:
                    print(torch.max(torch.abs(obs[:, term_begin_idx[5]:term_end_idx[5]])))
                # obj_state = obs[0][term_begin_idx[6]:term_end_idx[6]]
                # print("object_position: \n", torch.cat([obj_state[idx*obj_state_dim+obj_term_begin_idx[0]:idx*obj_state_dim+obj_term_end_idx[0]] for idx in range(obj_state_his_len)]))
                # print("object_lin_vel: \n", torch.cat([obj_state[idx*obj_state_dim+obj_term_begin_idx[1]:idx*obj_state_dim+obj_term_end_idx[1]] for idx in range(obj_state_his_len)]))
                # print("object_orientation: \n", torch.cat([obj_state[idx*obj_state_dim+obj_term_begin_idx[2]:idx*obj_state_dim+obj_term_end_idx[2]] for idx in range(obj_state_his_len)]))
                # print("object_ang_vel: \n", torch.cat([obj_state[idx*obj_state_dim+obj_term_begin_idx[3]:idx*obj_state_dim+obj_term_end_idx[3]] for idx in range(obj_state_his_len)]))
                # print("last object state: \n", obj_state[-obj_state_dim:])
                # print("Done: ", dones)
                # input("Press Enter to continue...")
            if mask_obs:
                # position
                obs[:, get_term_idx(6):get_term_idx(7)] = torch.tensor([0.0, 0.0, 0.143]*6, device=obs.device)
                # lin vel
                obs[:, get_term_idx(7):get_term_idx(8)] = torch.tensor([0.0, 0.0, 0.0]*6, device=obs.device)
                # orientation
                obs[:, get_term_idx(8):get_term_idx(9)] = torch.tensor([0.0, 0.0, 0.0, 1.0]*6, device=obs.device)
                # ang vel
                obs[:, get_term_idx(9):get_term_idx(10)] = torch.tensor([0.0, 0.0, 0.0]*6, device=obs.device)

            # add encoder observations to the policy
            # if actor_critic_class == "ActorCriticEncoder":
            #     encoder_obs = extras["observations"]["encoder"]
            #     obs = torch.cat((obs, encoder_obs), dim=1)

            # agent stepping
            actions = policy(obs)
            # print("Actions: ", actions[0])
            # input("Press Enter to continue...")
            # env stepping
            # obs, _, dones, extras = env.step(actions)
            next_obs, _, dones, extras = env.step(actions)
            obs = next_obs["policy"]
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
