#!/usr/bin/env python3
"""Play Go1 policy with web streaming enabled for remote visualization."""

import argparse
from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Play Go1 with web streaming")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Unitree-Go1-Play-v0", help="Task name")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")

# Add AppLauncher args (this adds --enable_livestream automatically)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch app with livestream
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of imports after app launch."""

import gymnasium as gym
import torch
import os

from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner


def main():
    """Play policy with web streaming."""
    
    # Parse environment config
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.cpu
    )
    
    # Create environment
    print(f"[INFO] Creating environment: {args_cli.task}")
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    
    # Load checkpoint
    print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
    if not os.path.exists(args_cli.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args_cli.checkpoint}")
    
    # Get agent config
    agent_cfg_entry_point = gym.spec(args_cli.task).kwargs.get("rsl_rl_cfg_entry_point")
    if isinstance(agent_cfg_entry_point, str):
        module_path, class_name = agent_cfg_entry_point.rsplit(":", 1)
        module = __import__(module_path, fromlist=[class_name])
        agent_cfg = getattr(module, class_name)()
    else:
        agent_cfg = agent_cfg_entry_point()
    
    # Create runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
    
    # Load policy
    runner.load(args_cli.checkpoint)
    print("[INFO] Policy loaded successfully")
    
    # Print streaming info
    print("\n" + "="*60)
    print("🌐 WEB STREAMING ENABLED")
    print("="*60)
    print("Access the visualization at:")
    print("  http://localhost:8211/streaming/webrtc-client")
    print("\nIf on remote server, create SSH tunnel:")
    print("  ssh -L 8211:localhost:8211 maxime@YOUR_SERVER")
    print("="*60 + "\n")
    
    # Run policy
    print("[INFO] Running policy... Press Ctrl+C to stop")
    
    obs, _ = env.get_observations()
    
    step_count = 0
    try:
        while simulation_app.is_running():
            # Get actions from policy
            with torch.no_grad():
                actions = runner.alg.act(obs, deterministic=True)
            
            # Step environment
            obs, _, _, _ = env.step(actions)
            
            step_count += 1
            if step_count % 500 == 0:
                print(f"[INFO] Steps: {step_count}")
    
    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")
    
    # Cleanup
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
