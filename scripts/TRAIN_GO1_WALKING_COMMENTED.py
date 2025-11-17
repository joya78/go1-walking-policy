#!/usr/bin/env python3
"""
Train a walking policy for Unitree Go1 robot using RSL RL PPO.

This script trains a neural network policy to control the Go1 robot to track
velocity commands while walking on rough terrain using Proximal Policy Optimization (PPO).

Key steps:
1. Parse command-line arguments
2. Initialize Isaac Sim app
3. Load environment and agent configurations
4. Create parallel training environments
5. Run PPO training loop
6. Save checkpoints periodically

Training Time: ~60 minutes for 1500 iterations on 4 GPUs
Output: Neural network weights saved as .pt files (PyTorch format)
"""

import argparse

from isaaclab.app import AppLauncher

# ============================================================================
# COMMAND-LINE ARGUMENTS
# ============================================================================

# Create argument parser for command-line configuration
parser = argparse.ArgumentParser(description="Train Unitree Go1 walking policy with RSL RL PPO.")

# ---- Environment Arguments ----
# Number of parallel environments to simulate
parser.add_argument(
    "--num_envs",
    type=int,
    default=None,
    help="Number of environments to simulate (default: 4096 for training)"
)

# Task name: registered environment in Isaac Lab
# This maps to the environment config and agent config
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Velocity-Rough-Unitree-Go1-v0",
    help="Name of the task / environment to train on"
)

# ---- Training Arguments ----
# Random seed for reproducibility
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Random seed for environment and policy initialization"
)

# Number of training iterations
# Each iteration: collect rollouts + update policy
# Runtime: ~2.4s per iteration on 4 GPUs with 4096 envs
parser.add_argument(
    "--max_iterations",
    type=int,
    default=None,
    help="Maximum RL training iterations (default: 1500)"
)

# Resume from checkpoint
parser.add_argument(
    "--resume",
    action="store_true",
    help="Resume training from the latest checkpoint"
)

# Specific checkpoint file to resume from
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Path to specific checkpoint file to resume from"
)

# AppLauncher arguments (Omniverse/NVIDIA specific)
# Includes: --headless (no GUI), --cpu (force CPU), --device (CUDA device)
AppLauncher.add_app_launcher_args(parser)

# Parse all arguments
args_cli = parser.parse_args()

# Launch the Omniverse app (Isaac Sim)
# This initializes NVIDIA physics, rendering, and simulation infrastructure
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ============================================================================
# IMPORTS (must come after app initialization)
# ============================================================================

import gymnasium as gym
import os
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner  # PPO trainer from RSL-RL library

from isaaclab.envs import ManagerBasedRLEnvCfg  # Config class for Isaac Lab environments
from isaaclab.utils.dict import print_dict  # Pretty-print configuration dicts

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # Wrappers

import isaaclab_tasks  # noqa: F401  # Register Isaac Lab tasks
from isaaclab_tasks.utils import parse_env_cfg  # Load environment config from registry

# ============================================================================
# GPU OPTIMIZATION
# ============================================================================
# Enable faster (but less precise) GPU math for speed

# TensorFloat32: trades precision for speed on Tensor Cores
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Determinism disabled for speed (non-deterministic but faster)
torch.backends.cudnn.deterministic = False

# Enable autotune for best kernel selection on specific GPU
torch.backends.cudnn.benchmark = False


def main():
    """
    Main training function.
    
    This function:
    1. Loads configuration (environment + agent)
    2. Overrides configs from command line
    3. Creates training environment(s)
    4. Initializes PPO trainer
    5. Runs training loop
    """

    # ========================================================================
    # 1. LOAD CONFIGURATIONS
    # ========================================================================

    print("[INFO] Loading environment and agent configurations...")

    # Parse environment configuration from task registry
    # This loads the Go1 environment config with 4096 parallel environments
    env_cfg = parse_env_cfg(
        args_cli.task,  # Task name: "Isaac-Velocity-Rough-Unitree-Go1-v0"
        device=args_cli.device,  # GPU device (e.g., "cuda:0")
        num_envs=args_cli.num_envs,  # Override environment count if provided
        use_fabric=not args_cli.cpu  # Use NVIDIA Fabric for multi-GPU (if not CPU-only)
    )

    # Get the agent configuration entry point from task registry
    # Each task specifies which agent config to use
    agent_cfg_entry_point = gym.spec(args_cli.task).kwargs.get("rsl_rl_cfg_entry_point")

    if agent_cfg_entry_point is None:
        raise ValueError(f"Task {args_cli.task} does not have an RSL-RL config entry point.")

    # Load agent configuration (PPO hyperparameters, network architecture, etc.)
    if isinstance(agent_cfg_entry_point, str):
        # Dynamically import config class from string path
        # Format: "module.path:ClassName"
        module_path, class_name = agent_cfg_entry_point.rsplit(":", 1)
        module = __import__(module_path, fromlist=[class_name])
        agent_cfg_class = getattr(module, class_name)
        agent_cfg = agent_cfg_class()
    else:
        # Config is a callable (class)
        agent_cfg = agent_cfg_entry_point()

    print("[INFO] Configurations loaded successfully!")

    # ========================================================================
    # 2. OVERRIDE CONFIGURATIONS FROM COMMAND LINE
    # ========================================================================

    # Override seed if provided (for reproducibility)
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
        print(f"[INFO] Seed overridden to: {args_cli.seed}")

    # Override max iterations if provided
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
        print(f"[INFO] Max iterations overridden to: {args_cli.max_iterations}")

    # Override environment parameters
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
        print(f"[INFO] Number of environments: {args_cli.num_envs}")

    # Set random seed for both environment and agent
    env_cfg.seed = agent_cfg.seed

    # Set device for simulation
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # ========================================================================
    # 3. SETUP LOGGING DIRECTORY
    # ========================================================================

    # Create directory structure for saving training results
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging directory: {log_root_path}")

    # Determine if resuming or starting fresh
    resume_path = None  # Path to checkpoint file to load

    if args_cli.checkpoint:
        # User specified exact checkpoint file
        resume_path = args_cli.checkpoint
        print(f"[INFO] Resuming from specified checkpoint: {resume_path}")
        # Use the directory containing the checkpoint
        log_dir = os.path.dirname(resume_path)

    elif args_cli.resume:
        # Auto-find latest checkpoint
        if os.path.exists(log_root_path):
            # Find most recent training run
            run_dirs = sorted([d for d in os.listdir(log_root_path)
                             if os.path.isdir(os.path.join(log_root_path, d))])

            if run_dirs:
                latest_run = run_dirs[-1]  # Most recent timestamp
                checkpoint_dir = os.path.join(log_root_path, latest_run)

                # Find latest checkpoint file
                checkpoints = [f for f in os.listdir(checkpoint_dir)
                             if f.startswith("model_") and f.endswith(".pt")]

                if checkpoints:
                    # Sort by iteration number and pick latest
                    latest_checkpoint = sorted(
                        checkpoints,
                        key=lambda x: int(x.split("_")[1].split(".")[0])
                    )[-1]

                    resume_path = os.path.join(checkpoint_dir, latest_checkpoint)
                    log_dir = checkpoint_dir
                    print(f"[INFO] Auto-resuming from: {resume_path}")
                else:
                    # No checkpoints found, start fresh
                    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    log_dir = os.path.join(log_root_path, log_dir)
                    print("[WARNING] No checkpoints found, starting fresh training")
            else:
                # No previous runs, start fresh
                log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                log_dir = os.path.join(log_root_path, log_dir)
                print("[WARNING] No previous runs found, starting fresh training")
        else:
            # Log directory doesn't exist, create fresh
            log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_dir = os.path.join(log_root_path, log_dir)
            print("[INFO] Creating fresh training log directory")

    else:
        # Fresh training (default)
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join(log_root_path, log_dir)
        print("[INFO] Starting fresh training run")

    # ========================================================================
    # 4. PRINT CONFIGURATIONS
    # ========================================================================

    print("\n" + "="*80)
    print("ENVIRONMENT CONFIGURATION")
    print("="*80)
    print_dict(env_cfg.to_dict(), nesting=4)

    print("\n" + "="*80)
    print("AGENT CONFIGURATION (PPO Hyperparameters)")
    print("="*80)
    print_dict(agent_cfg.to_dict(), nesting=4)

    # ========================================================================
    # 5. CREATE TRAINING ENVIRONMENT
    # ========================================================================

    print("\n[INFO] Creating Isaac Lab environment...")
    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Num Envs: {env_cfg.scene.num_envs}")
    print(f"[INFO] Device: {env_cfg.sim.device}")

    # Create gym environment using registry
    # This instantiates Isaac Sim scenes with 4096 parallel environments
    env = gym.make(args_cli.task, cfg=env_cfg)

    # Wrap environment for RSL-RL compatibility
    # Converts Isaac Lab env format to RSL-RL expected format
    env = RslRlVecEnvWrapper(env)

    print("[INFO] Environment created successfully!")
    print(f"[INFO] Observation space: {env.observation_space}")
    print(f"[INFO] Action space: {env.action_space}")

    # ========================================================================
    # 6. CREATE PPO TRAINER AND RUN TRAINING
    # ========================================================================

    print("\n[INFO] Initializing PPO trainer...")

    # Create OnPolicyRunner (PPO trainer from RSL-RL)
    # This handles:
    # - Collecting rollouts from environment
    # - Computing advantages and returns
    # - Training policy and value network
    # - Saving checkpoints
    runner = OnPolicyRunner(
        env,                          # Training environment
        agent_cfg.to_dict(),          # Agent hyperparameters
        log_dir=log_dir,              # Where to save checkpoints/logs
        device=agent_cfg.device       # GPU device for training
    )

    # Load checkpoint if resuming training
    if resume_path:
        print(f"[INFO] Loading checkpoint: {resume_path}")
        runner.load(resume_path)
        print("[INFO] Checkpoint loaded!")

    # ====== TRAINING LOOP ======
    print("\n" + "="*80)
    print(f"[INFO] Starting PPO training for {agent_cfg.max_iterations} iterations...")
    print(f"[INFO] Saving checkpoints to: {log_dir}")
    print("="*80 + "\n")

    # Run the training loop
    # This will:
    # 1. Sample actions from policy
    # 2. Step environments
    # 3. Collect rewards and trajectories
    # 4. Compute advantages
    # 5. Update policy with PPO loss
    # 6. Save checkpoint every N iterations
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,  # Total training steps
        init_at_random_ep_len=True  # Randomize episode start for stability
    )

    print("\n" + "="*80)
    print("[INFO] Training completed!")
    print(f"[INFO] Final checkpoints saved to: {log_dir}")
    print("="*80)

    # ========================================================================
    # 7. CLEANUP
    # ========================================================================

    print("[INFO] Closing environment...")
    env.close()

    print("[INFO] Training script finished successfully!")


if __name__ == "__main__":
    """
    Entry point for the training script.
    
    Wraps main() in try-except for graceful error handling.
    Ensures simulation app is closed on exit.
    """

    try:
        # Run main training function
        main()

    except Exception as e:
        # Print error details if training fails
        print(f"\n[ERROR] Training failed with error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Always close the simulation app on exit
        print("[INFO] Closing Isaac Sim application...")
        simulation_app.close()
        print("[INFO] Bye!")
