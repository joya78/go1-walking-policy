#!/usr/bin/env python3
"""Script to capture images of the robot executing the trained policy."""

import argparse
import os
from pathlib import Path

# Isaac Sim imports
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Capture images of the robot executing the trained policy")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
parser.add_argument("--num_steps", type=int, default=200, help="Number of steps to run")
parser.add_argument("--save_interval", type=int, default=10, help="Save image every N steps")
parser.add_argument("--output_dir", type=str, default="/home/maxime/my_go1_project/videos/robot_images", help="Output directory")
args_cli, app_launcher_args = parser.parse_known_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli=app_launcher_args)
simulation_app = app_launcher.app

import torch
import carb
import omni.kit.viewport.utility as viewport_utils
from omni.isaac.core.utils.viewports import set_camera_view

# Import Isaac Lab
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
import isaaclab_tasks
from rsl_rl.runners import OnPolicyRunner

def main():
    # Setup output directory
    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Images will be saved to: {output_dir}")
    
    # Create environment
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        "Isaac-Velocity-Rough-Unitree-Go1-Play-v0",
        use_gpu=True,
        num_envs=1
    )
    env = isaaclab.envs.ManagerBasedRLEnv(cfg=env_cfg)
    
    # Load checkpoint
    print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
    agent_cfg = env_cfg.agent_cfg
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
    runner.load(args_cli.checkpoint)
    
    # Setup camera view
    print("[INFO] Setting up camera...")
    viewport = viewport_utils.get_active_viewport()
    set_camera_view(eye=[3.0, 3.0, 2.0], target=[0.0, 0.0, 0.5], viewport=viewport)
    
    # Reset environment
    obs, _ = env.reset()
    
    print(f"[INFO] Starting robot visualization and capture...")
    print(f"[INFO] Will save {args_cli.num_steps // args_cli.save_interval} images")
    
    # Run simulation and capture images
    for step in range(args_cli.num_steps):
        # Get action from policy
        with torch.no_grad():
            actions = runner.alg.act(obs, deterministic=True)
        
        # Step environment
        obs, _, _, _ = env.step(actions)
        
        # Capture image at intervals
        if step % args_cli.save_interval == 0:
            image_path = output_dir / f"robot_step_{step:04d}.png"
            
            # Capture viewport to file
            try:
                viewport_utils.capture_viewport_to_file(
                    viewport=viewport,
                    file_path=str(image_path)
                )
                print(f"[INFO] Saved image: {image_path.name}")
            except Exception as e:
                print(f"[WARNING] Failed to capture image at step {step}: {e}")
    
    print(f"[INFO] Capture complete! Images saved to {output_dir}")
    
    # Cleanup
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
