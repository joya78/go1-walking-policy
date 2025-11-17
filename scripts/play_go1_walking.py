#!/usr/bin/env python3
"""Play/test a trained walking policy for Unitree Go1 robot."""

import argparse

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Play trained Unitree Go1 walking policy.")
parser.add_argument("--num_envs", type=int, default=50, help="Number of environments to simulate")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Unitree-Go1-Play-v0", help="Name of the task")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
parser.add_argument("--video", action="store_true", default=False, help="Record video of the policy")
parser.add_argument("--video_length", type=int, default=500, help="Length of the video (in steps)")
parser.add_argument("--video_dir", type=str, default="videos", help="Directory to save videos")
parser.add_argument("--video_interval", type=int, default=250, help="Interval between video frames (lower=smoother, higher=faster)")

# Append AppLauncher CLI args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()
# Launch omniverse app
# Note: If you have a display/X11 server, set headless=False for GUI
# For remote/headless systems, keep headless=True (or pass --headless flag)
if not hasattr(args_cli, 'headless') or args_cli.headless is None:
    args_cli.headless = True  # Default to headless on systems without display
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import os
from datetime import datetime

from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg
from rsl_rl.runners import OnPolicyRunner


def _load_agent_cfg(task: str):
    """Load the RSL-RL agent config from the gym registry entry point.

    Isaac Lab tasks expose `rsl_rl_cfg_entry_point` under the gym spec.
    This returns a config class which we instantiate here.
    """
    entry = gym.spec(task).kwargs.get("rsl_rl_cfg_entry_point")
    if entry is None:
        raise ValueError(f"Task {task} does not define 'rsl_rl_cfg_entry_point'.")
    if isinstance(entry, str):
        module_path, class_name = entry.rsplit(":", 1)
        module = __import__(module_path, fromlist=[class_name])
        cfg_cls = getattr(module, class_name)
        return cfg_cls()
    return entry()


def main():
    """Play with trained RSL RL agent."""
    # Parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.cpu
    )

    # Create Isaac Lab environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # Setup video recording if requested (using gym's RecordVideo wrapper)
    video_env = None
    if args_cli.video:
        video_dir = os.path.abspath(args_cli.video_dir)
        os.makedirs(video_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name_prefix = f"go1_policy_{timestamp}"
        print(f"[INFO] Video recording enabled. Saving to: {video_dir}/{video_name_prefix}")
        
        # Gym's RecordVideo wrapper needs render() support
        # We'll manually capture and save frames instead
        from PIL import Image
        import numpy as np
        video_frames = []
        video_enabled = True
    else:
        video_enabled = False
    
    # Wrap for RSL-RL
    env = RslRlVecEnvWrapper(env)
    
    # Check if checkpoint is provided
    if args_cli.checkpoint is None:
        print("[WARNING] No checkpoint provided. Looking for latest checkpoint...")
        # Try to find the latest checkpoint
        log_root = "logs/rsl_rl/unitree_go1_rough"
        if os.path.exists(log_root):
            run_dirs = [d for d in os.listdir(log_root) if os.path.isdir(os.path.join(log_root, d))]
            if run_dirs:
                latest_run = sorted(run_dirs)[-1]
                checkpoint_dir = os.path.join(log_root, latest_run)
                checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("model_") and f.endswith(".pt")]
                if checkpoints:
                    latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("_")[1].split(".")[0]))[-1]
                    args_cli.checkpoint = os.path.join(checkpoint_dir, latest_checkpoint)
                    print(f"[INFO] Using checkpoint: {args_cli.checkpoint}")
                else:
                    print("[ERROR] No checkpoints found in the latest run directory.")
                    env.close()
                    return
            else:
                print(f"[ERROR] No run directories found in {log_root}")
                env.close()
                return
        else:
            print(f"[ERROR] Log directory {log_root} not found. Please provide checkpoint path.")
            env.close()
            return
    
    # Load the trained agent
    print(f"[INFO] Loading checkpoint from: {args_cli.checkpoint}")
    agent_cfg = _load_agent_cfg(args_cli.task)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    inference_policy = runner.get_inference_policy()
    
    # Play the policy
    print("[INFO] Starting policy playback...")
    print("[INFO] Press Ctrl+C to stop")
    
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    
    step_count = 0
    episode_count = 0
    total_reward = 0.0
    episode_reward = 0.0
    episode_length = 0
    
    # Video recording setup
    if video_enabled:
        try:
            import imageio
            video_writer = imageio.get_writer(
                os.path.join(video_dir, f"{video_name_prefix}.mp4"),
                fps=50 // args_cli.video_interval,
                codec='libx264',
                pixelformat='yuv420p'
            )
            print(f"[INFO] Video writer initialized: {video_name_prefix}.mp4")
        except ImportError:
            print("[WARNING] imageio not installed. Video recording disabled.")
            print("[INFO] Install with: pip install imageio imageio-ffmpeg")
            video_enabled = False
    
    try:
        while simulation_app.is_running():
            # Get action from policy
            with torch.no_grad():
                actions = inference_policy(obs)

            # Step the environment (RSL-RL vec wrapper returns 4-tuple)
            obs, rewards, dones, infos = env.step(actions)
            
            # Capture frame for video if enabled
            if video_enabled and step_count % args_cli.video_interval == 0:
                try:
                    # Try to get render from unwrapped env
                    frame = env.unwrapped.sim.render()
                    if frame is not None:
                        video_writer.append_data(frame)
                except Exception as e:
                    pass  # Skip frame if render fails
            
            episode_reward += rewards.mean().item()
            episode_length += 1
            step_count += 1
            
            # Check for episode completion
            if dones.any():
                episode_count += dones.sum().item()
                avg_reward = episode_reward / episode_length if episode_length > 0 else 0
                print(f"[INFO] Episode {episode_count} | Steps: {episode_length} | Avg Reward: {avg_reward:.3f}")
                total_reward += episode_reward
                episode_reward = 0.0
                episode_length = 0
            
            # Stop after video length if recording
            if video_enabled and step_count >= args_cli.video_length:
                print(f"[INFO] Video recording complete ({args_cli.video_length} steps)")
                video_writer.close()
                print(f"[INFO] Video saved to: {os.path.join(video_dir, f'{video_name_prefix}.mp4')}")
                break
                
    except KeyboardInterrupt:
        print("\n[INFO] Playback interrupted by user")
        if video_enabled:
            video_writer.close()
            print(f"[INFO] Video saved to: {os.path.join(video_dir, f'{video_name_prefix}.mp4')}")
    
    # Print final statistics
    if step_count > 0:
        print(f"\n[INFO] Final Statistics:")
        print(f"  Total Steps: {step_count}")
        print(f"  Episodes Completed: {episode_count}")
        print(f"  Average Reward: {total_reward / step_count:.3f}")
    
    # Close the environment
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] Playback failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
