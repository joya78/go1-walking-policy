#!/usr/bin/env python3
"""Visualize Go1 robot by saving camera images during policy execution."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize Go1 with image capture")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Unitree-Go1-Play-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_steps", type=int, default=500, help="Number of steps to run")
parser.add_argument("--save_interval", type=int, default=10, help="Save image every N steps")
parser.add_argument("--output_dir", type=str, default="/home/maxime/my_go1_project/videos/robot_images")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import os
from pathlib import Path
import numpy as np
from PIL import Image

from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

# Import Isaac Sim rendering
import omni.replicator.core as rep


def main():
    """Run policy and save camera images."""
    
    # Create output directory
    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Images will be saved to: {output_dir}")
    
    # Create environment
    print(f"[INFO] Creating environment: {args_cli.task}")
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    
    # Load policy
    print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
    agent_cfg_entry_point = gym.spec(args_cli.task).kwargs.get("rsl_rl_cfg_entry_point")
    if isinstance(agent_cfg_entry_point, str):
        module_path, class_name = agent_cfg_entry_point.rsplit(":", 1)
        module = __import__(module_path, fromlist=[class_name])
        agent_cfg = getattr(module, class_name)()
    else:
        agent_cfg = agent_cfg_entry_point()
    
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
    runner.load(args_cli.checkpoint)
    
    # Setup camera with replicator
    print("[INFO] Setting up camera...")
    camera = rep.create.camera(
        position=(3, 3, 2),
        look_at=(0, 0, 0.5)
    )
    
    # Render product
    render_product = rep.create.render_product(camera, (1280, 720))
    
    # Writer to save images
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir=str(output_dir),
        rgb=True,
        render_product=render_product
    )
    
    print(f"[INFO] Running policy for {args_cli.num_steps} steps...")
    print(f"[INFO] Saving every {args_cli.save_interval} steps")
    
    obs, _ = env.get_observations()
    images_saved = 0
    
    try:
        for step in range(args_cli.num_steps):
            # Get action from policy
            with torch.no_grad():
                actions = runner.alg.act(obs, deterministic=True)
            
            # Step environment
            obs, _, _, _ = env.step(actions)
            
            # Save image at intervals
            if step % args_cli.save_interval == 0:
                rep.orchestrator.step(rt_subframes=4)
                writer.write()
                images_saved += 1
                
                if step % 100 == 0:
                    print(f"  📸 Step {step}/{args_cli.num_steps} - Images saved: {images_saved}")
        
        print(f"\n✅ Done! Saved {images_saved} images to {output_dir}")
        print(f"\n📊 To create animation:")
        print(f"  cd {output_dir}")
        print(f"  ffmpeg -framerate 10 -pattern_type glob -i '*.png' -c:v libx264 go1_animation.mp4")
        
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    
    # Cleanup
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
