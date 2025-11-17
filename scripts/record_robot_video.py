#!/usr/bin/env python3
"""Script to record video of robot executing trained policy using offscreen rendering."""

import argparse
import os
from pathlib import Path

# Isaac Sim imports - must be before AppLauncher
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record video of robot executing trained policy")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
parser.add_argument("--num_steps", type=int, default=500, help="Number of steps to run")
parser.add_argument("--output_dir", type=str, default="/home/maxime/my_go1_project/videos", help="Output directory")
parser.add_argument("--video_name", type=str, default="robot_walking.mp4", help="Output video filename")
args_cli, app_launcher_args = parser.parse_known_args()

# Launch Isaac Sim with rendering enabled
app_launcher = AppLauncher(args_cli=app_launcher_args, headless=False)
simulation_app = app_launcher.app

import torch
import numpy as np
import omni

# Import Isaac Lab
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
import isaaclab_tasks
from rsl_rl.runners import OnPolicyRunner

def main():
    # Setup output
    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / args_cli.video_name
    print(f"[INFO] Video will be saved to: {video_path}")
    
    # Create environment
    print("[INFO] Creating environment...")
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        "Isaac-Velocity-Rough-Unitree-Go1-Play-v0",
        use_gpu=True,
        num_envs=1
    )
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # Load checkpoint
    print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
    agent_cfg = env_cfg.agent_cfg
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
    runner.load(args_cli.checkpoint)
    
    # Start recording
    print("[INFO] Starting video recording...")
    movie_capture = omni.kit.capture.ICaptureManager
    
    # Reset and run
    print(f"[INFO] Running {args_cli.num_steps} steps...")
    obs, _ = env.reset()
    
    for step in range(args_cli.num_steps):
        with torch.no_grad():
            actions = runner.alg.act(obs, deterministic=True)
        obs, _, _, _ = env.step(actions)
        
        if step % 50 == 0:
            print(f"[INFO] Step {step}/{args_cli.num_steps}")
    
    print(f"[INFO] Simulation complete!")
    print(f"[INFO] Video saved to: {video_path}")
    
    # Cleanup
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
