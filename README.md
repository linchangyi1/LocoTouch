# Perceptive Learning for Legged Robots in IsaacLab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0-green)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.2.1-green)](https://isaac-sim.github.io/IsaacLab)
[![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04-red)](https://docs.python.org/3/whatsnew/3.10.html)



### Table of Contents
1. [Overview](#overview)  
2. [Key Features](#key-features)
3. [Installation](#installation)  
4. [Training, Play, and Deployment](#training-play-and-deployment)
5. [Known Issues](#known-issues)
6. [Notes](#notes)




## Overview
This repository provides code for training **perceptive** policies for legged robots using the **manager**-based workflow in **IsaacLab**.
Network architectures, distillation methods, and other configurations can be **easily customized** through parameters.
The learned policies can be tested with **MoCap** or deployed on an onboard **Mac Mini** computer.

Please note that the official implementation of LocoTouch is built on IsaacSim 4.5 and IsaacLab 1.40. To reproduce LocoTouch with Ubuntu 20.04, please refer to the `ubuntu-20` branch.


## Key Features
- **[LocoTouch](https://arxiv.org/abs/2505.23175)** Implementations
  - [x] Quadrupedal Transport Policy with Tactile Sensing
    <img src="rm_figs/locotouch_pipeline.png" width="100%">
  - [x] Adaptive Symmetric Locomotion Gait (via Symmetricity Reward Function)
    <img src="rm_figs/symmetric_gait.png" width="100%">


- **Flexible Network Architectures** for RL/IL
  - [x] MLP Policy w/wo MLP/CNN Encoder, w/wo RNN Module
    <img src="rm_figs/models.png" width="80%">
  - [ ] Transformer Policy
  - [ ] Diffusion Policy


- **Teacher-Student** Pipelines
  - [x] "[RMA](https://arxiv.org/pdf/2107.04034)" (Latent Supervision)
  - [x] "[Monolithic](https://arxiv.org/pdf/2211.07638)" (Action Supervision)
  <!-- <img src="rm_figs/rma_monolithic.png" width="80%" comment='copy from https://arxiv.org/pdf/2211.07638'> -->
  - [ ] "[ROA](https://arxiv.org/pdf/2210.10044)" (Regularized Online Latent Supervision)

- **[Policy Deployment](https://github.com/linchangyi1/Go1-Policy-Deployment)**
  - [x] Signal Reading and Processing for Distributed Tactile Sensors
  - [x] MoCap-Robot Communication via NaNet and ROS
  - [x] IsaacLab-compatible depoyment code for Locomotion, State-Based Transport, and Tactile-Aware Transport polcies
  - [x] Mac Mini Setup with Custom-Designed Shell ([MacMini-for-Onboard-Robotics](https://github.com/linchangyi1/MacMini-for-Onboard-Robotics))
    <img src="rm_figs/full_back_example.png" width="100%">


- **Evaluated Perception Modalities**
  - [x] Proprioception (Motor / IMU)
  - [x] Tactile Sensing
  - [ ] Visual Sensing (Depth / RGB)
  - [ ] Other Modalities (e.g., LiDAR)

Please note that the unimplemented features are not part of [LocoTouch](https://arxiv.org/abs/2505.23175), and there is no estimated timeline for their release.

## Installation

### System and Software requirements:
- Ubuntu 22.04 (x64)
- GelForce RTX 4090 GPU (recommended)
- NVIDIA GPU Driver 570.169
- NVIDIA CUDA 12.8

#### Install IsaacSim and IsaacLab:
- Follow the [guide](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html) to install IsaacSim 5.0.
- Follow the [guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html) to install IsaacLab 2.2.1 inside a Conda environment.



#### Install LocoTouch:
Using a python interpreter where IsaacLab is installed, install LocoTouch and Loco_RL:
```bash
cd LocoTouch
pip install -e .
cd loco_rl
pip install -e .
cd ..
```

Verify the installation by playing the teacher and student policies:
  - Teacher Policy:
    ```bash
    python locotouch/scripts/play.py --task Isaac-RandCylinderTransportTeacher-LocoTouch-Play-v1 --num_envs=20 --load_run=2025-09-01_21-03-58
    ```
  - Student Policy:
    ```bash
    python locotouch/scripts/distill.py --task Isaac-RandCylinderTransportStudent_SingleBinaryTac_CNNRNN_Mon-LocoTouch-Play-v1 --num_envs=20 --log_dir_distill=2025-09-02_23-27-14 --checkpoint_distill=model_7.pt
    ```


## Training, Play, and Deployment
#### Locomotion (Optional)
- RL Training:
  ```bash
  python locotouch/scripts/train.py --task Isaac-Locomotion-LocoTouch-v1 --num_envs=4096 --headless
  ```
- Play:
  ```bash
  python locotouch/scripts/play.py --task Isaac-Locomotion-LocoTouch-Play-v1 --num_envs=20
  ```
- Deployment:
  Follow the guide in [Go1-Policy-Deployment](https://github.com/linchangyi1/Go1-Policy-Deployment) for installation, and run:
  ```bash
  python teleoperation/joystick.py
  ```
  ```bash
  python deploy/locomotion.py
  ```

#### State-Based Object Transport (Teacher Policy)
- RL Training (recommend 15k+ iterations):
  ```bash
  python locotouch/scripts/train.py --task Isaac-RandCylinderTransportTeacher-LocoTouch-v1 --num_envs=4096 --headless
  ```
- Play:
  ```bash
  python locotouch/scripts/play.py --task Isaac-RandCylinderTransportTeacher-LocoTouch-Play-v1 --num_envs=20
  ```
- Deployment:
  Follow the guide in [Go1-Policy-Deployment](https://github.com/linchangyi1/Go1-Policy-Deployment) for installation, and run:
  ```bash
  python teleoperation/joystick.py
  ```
  ```bash
  python mocap/run_optitrack.py
  ```
  ```bash
  python deploy/transport_teacher.py
  ```
  <img src="rm_figs/teacher_sim.gif" width="48%">
  <img src="rm_figs/teacher_real.gif" width="48%">



#### Tactile-Aware Object Transport (Student Policy)
- Distillation:
  ```bash
  python locotouch/scripts/distill.py --task Isaac-RandCylinderTransportStudent_SingleBinaryTac_CNNRNN_Mon-LocoTouch-v1 --training --num_envs=405 --headless --load_run=2025-09-01_21-03-58
  ```

  Additional options are supported, e.g.:
  ```bash
  python locotouch/scripts/distill.py --task Isaac-RandCylinderTransportStudent_SingleBinaryTac_CNNRNN_Mon-LocoTouch-v1 --training --num_envs=405 --headless --load_run=2025-09-01_21-03-58 --checkpoint=model_15000.pt --headless --distill_lr=0.0005
  ```

- Play (replace with your own log_dir_distill folder):
  ```bash
  python locotouch/scripts/distill.py --task Isaac-RandCylinderTransportStudent_SingleBinaryTac_CNNRNN_Mon-LocoTouch-Play-v1 --num_envs=20 --log_dir_distill=2025-09-02_23-27-14 --checkpoint_distill=model_7.pt
  ```
- Deployment:
  Follow the guide in [Go1-Policy-Deployment](https://github.com/linchangyi1/Go1-Policy-Deployment) for installation, and run:
  ```bash
  python teleoperation/joystick.py
  ```
  ```bash
  python tactile_sensing/run_tactile_sensing.py
  ```
  ```bash
  python deploy/transport_student.py
  ```
  <img src="rm_figs/student_sim.gif" width="48%">
  <img src="rm_figs/student_real.gif" width="48%">



## Known Issues
- Python is not installed:
  ```bash
  conda install python=3.11
  ```
- Error occurs after running "./isaaclab.sh --install":
  ```bash
  pip install --upgrade pip
  ```
- Visualization error occurs after running "python source/standalone/tutorials/00_sim/create_empty.py":
  ```bash
  conda remove --force xorg-libxcb xorg-libx11 xorg-libxext libxcb libxkbcommon mesa-glu libglvnd
  ```


## Notes
- Mitigating from IsaacLab 1.40 to 2.0:
  omni.isaac.lab. -> isaaclab.
  omni.isaac.lab_assets -> isaaclab_assets
  omni.isaac.lab_tasks -> isaaclab_tasks
  omni.isaac.lab_tasks.utils.wrappers.rsl_rl -> isaaclab_rl.rsl_rl
  obs, extras = env.get_observations() ->
    ```bash
    env_obs = env.get_observations()
    obs = env_obs["policy"]
    extras = {"observations": {k: v for k, v in env_obs.items()}}
    ```
  obs, rwd, dones, extras = self.env.step(action) ->
    ```bash
    next_obs, _, dones, extras = self.env.step(action)
    extras = {"observations": {k: v for k, v in next_obs.items()}}
    ```
  quat_rotate_inverse -> quat_apply_inverse
  quat_rotate -> quat_apply


## VScode Setting
Press `ctrl` + `shift` + `p`; choose Open Workspace Settings; modify and paste the following settings:
```bash
{
    "python.analysis.extraPaths": [
        "${workspaceFolder}/_isaac_sim",
        "${workspaceFolder}/../IsaacLab/source/isaaclab_assets",
        "${workspaceFolder}/../IsaacLab/source/isaaclab_rl",
        "${workspaceFolder}/../IsaacLab/source/isaaclab",
        "${workspaceFolder}/../IsaacLab/source/isaaclab_tasks",
        "${workspaceFolder}/../IsaacLab/source/isaaclab_mimic"
    ]
}
```

## Reference
```bibtex
@article{lin2025locotouch,
  title={LocoTouch: Learning Dynamic Quadrupedal Transport with Tactile Sensing},
  author={Lin, Changyi and Song, Yuxin Ray and Huo, Boda and Yu, Mingyang and Wang, Yikai and Liu, Shiqi and Yang, Yuxiang and Yu, Wenhao and Zhang, Tingnan and Tan, Jie and others},
  journal={arXiv preprint arXiv:2505.23175},
  year={2025}
}
```
