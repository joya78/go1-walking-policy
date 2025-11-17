#!/usr/bin/env python3
"""Train a walking policy for Unitree Go1 robot using custom configuration."""

import sys
import os

# Add project config to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Train Unitree Go1 with custom configuration.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments")
parser.add_argument("--max_iterations", type=int, default=1500, help="Training iterations")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")

# Add AppLauncher args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows."""

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# Import custom configs
import config

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def main():
    """Train with custom config."""
    
    # Use custom environment
    task_name = "Isaac-Velocity-Rough-Unitree-Go1-Custom-v0"
    
    # Create environment
    env = gym.make(task_name, num_envs=args_cli.num_envs)
    env = RslRlVecEnvWrapper(env)
    
    # Get agent config
    agent_cfg = gym.spec(task_name).kwargs["rsl_rl_cfg_entry_point"]
    if isinstance(agent_cfg, str):
        module_path, class_name = agent_cfg.rsplit(":", 1)
        module = __import__(module_path, fromlist=[class_name])
        agent_cfg = getattr(module, class_name)()
    
    # Override from CLI
    agent_cfg.seed = args_cli.seed
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.resume = args_cli.resume
    
    # Create runner
    runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=args_cli.device)
    
    # Train
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    
    # Close
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
