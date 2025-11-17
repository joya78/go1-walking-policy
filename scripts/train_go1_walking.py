#!/usr/bin/env python3
"""Train a walking policy for Unitree Go1 robot using RSL RL.

This script is a simplified version of the official Isaac Lab RSL-RL training script,
adapted specifically for the Go1 robot.
"""

import argparse

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Train Unitree Go1 walking policy with RSL RL PPO.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Unitree-Go1-v0", help="Name of the task")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations")
parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
parser.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint to resume from")
# Add AppLauncher args (includes --headless, --cpu, --device, etc.)
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()
# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    """Train with RSL RL agent."""
    # Parse environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.cpu
    )
    
    # Get the agent config entry point from environment spec
    agent_cfg_entry_point = gym.spec(args_cli.task).kwargs.get("rsl_rl_cfg_entry_point")
    
    if agent_cfg_entry_point is None:
        raise ValueError(f"Task {args_cli.task} does not have an RSL-RL config entry point.")
    
    # Parse the agent config
    if isinstance(agent_cfg_entry_point, str):
        # Import the config module
        module_path, class_name = agent_cfg_entry_point.rsplit(":", 1)
        module = __import__(module_path, fromlist=[class_name])
        agent_cfg_class = getattr(module, class_name)
        agent_cfg = agent_cfg_class()
    else:
        agent_cfg = agent_cfg_entry_point()
    
    # Override configs from command line
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    
    # Override environment params
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    
    # Check for resume
    resume_path = None
    if args_cli.checkpoint:
        resume_path = args_cli.checkpoint
        print(f"[INFO] Resuming from specified checkpoint: {resume_path}")
    elif args_cli.resume:
        # Find latest checkpoint automatically
        if os.path.exists(log_root_path):
            run_dirs = sorted([d for d in os.listdir(log_root_path) 
                             if os.path.isdir(os.path.join(log_root_path, d))])
            if run_dirs:
                latest_run = run_dirs[-1]
                checkpoint_dir = os.path.join(log_root_path, latest_run)
                checkpoints = [f for f in os.listdir(checkpoint_dir) 
                             if f.startswith("model_") and f.endswith(".pt")]
                if checkpoints:
                    latest_checkpoint = sorted(checkpoints, 
                                             key=lambda x: int(x.split("_")[1].split(".")[0]))[-1]
                    resume_path = os.path.join(checkpoint_dir, latest_checkpoint)
                    print(f"[INFO] Auto-resuming from latest checkpoint: {resume_path}")
                    # Use the same log directory to continue
                    log_dir = checkpoint_dir
                else:
                    print("[WARNING] No checkpoints found, starting fresh training")
                    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    log_dir = os.path.join(log_root_path, log_dir)
            else:
                print("[WARNING] No previous runs found, starting fresh training")
                log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                log_dir = os.path.join(log_root_path, log_dir)
        else:
            print("[WARNING] No log directory found, starting fresh training")
            log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_dir = os.path.join(log_root_path, log_dir)
    else:
        # Fresh training
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join(log_root_path, log_dir)
    
    # Print configuration
    print("[INFO] Environment Configuration:")
    print_dict(env_cfg.to_dict(), nesting=4)
    print("[INFO] Agent Configuration:")
    print_dict(agent_cfg.to_dict(), nesting=4)
    
    # Create Isaac Lab environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # Wrap environment for RSL-RL
    env = RslRlVecEnvWrapper(env)
    
    print(f"[INFO] Starting training for {agent_cfg.max_iterations} iterations...")
    
    # Create the agent
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    
    # Load checkpoint if resuming
    if resume_path:
        print(f"[INFO] Loading checkpoint: {resume_path}")
        runner.load(resume_path)
    
    # Train
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    
    print(f"[INFO] Training completed! Checkpoints saved to: {log_dir}")
    
    # Close the environment
    env.close()


if __name__ == "__main__":
    # Run the main training function
    try:
        main()
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close the app
        simulation_app.close()
