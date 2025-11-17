#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
                    GO1 WALKING POLICY - HEADLESS TRAINING SCRIPT
═══════════════════════════════════════════════════════════════════════════════

This script trains a PPO policy for Go1 quadruped locomotion using Isaac Lab.

Key Features:
- Minimal dependencies (doesn't require full Isaac Sim installation)
- Headless execution (no GUI required)
- GPU acceleration with CUDA
- Real-time performance metrics
- Checkpoint saving and resuming

Usage:
    ./isaaclab.sh -p scripts/train_go1_headless.py --num_envs 512 --headless
    
Or with custom parameters:
    ./isaaclab.sh -p scripts/train_go1_headless.py \
        --num_envs 2048 \
        --max_iterations 5000 \
        --seed 42 \
        --headless

═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

# Add the repository root to path so we can import config
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner


def main(args):
    """Main training function."""
    
    print("\n" + "="*80)
    print("GO1 WALKING POLICY - PPO TRAINING")
    print("="*80)
    
    # Check GPU availability
    print(f"\n📊 System Information:")
    print(f"   GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"   PyTorch Version: {torch.__version__}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create experiment name and logging directory
    exp_name = f"go1_walk_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = Path(__file__).parent.parent / "logs" / exp_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Logging Directory: {log_dir}")
    
    # Create environment configuration
    from config.go1_walking_env_cfg import Go1WalkingEnvCfg
    
    env_cfg = Go1WalkingEnvCfg()
    env_cfg.num_envs = args.num_envs
    env_cfg.viewer.headless = args.headless
    
    # Override defaults with command-line arguments
    if args.seed is not None:
        env_cfg.seed = args.seed
    
    print(f"\n⚙️  Environment Configuration:")
    print(f"   Number of Environments: {env_cfg.num_envs}")
    print(f"   Headless Mode: {args.headless}")
    if hasattr(env_cfg, 'seed'):
        print(f"   Random Seed: {env_cfg.seed}")
    
    # Create environment
    print(f"\n🔨 Creating Environment...")
    try:
        env = gym.make("Isaac-Go1-Walking-v0", cfg=env_cfg)
        print(f"   ✓ Environment created successfully")
        print(f"   Observation Space: {env.observation_space}")
        print(f"   Action Space: {env.action_space}")
    except Exception as e:
        print(f"   ✗ Failed to create environment: {e}")
        print(f"   This likely means Isaac Lab is not fully initialized.")
        print(f"   Make sure to run this script using: isaaclab.sh -p scripts/train_go1_headless.py")
        return
    
    # Create agent configuration
    from rsl_rl.runners import OnPolicyRunnerCfg
    from config.agents.actor_critic import ActorCriticCfg
    
    agent_cfg = ActorCriticCfg()
    agent_cfg.device = str(device)
    
    # Wrap environment for RSL-RL
    from isaaclab.envs import RslRlVecEnvWrapper
    env = RslRlVecEnvWrapper(env)
    
    print(f"\n🤖 PPO Agent Configuration:")
    print(f"   Device: {device}")
    
    # Create trainer
    print(f"\n🎓 Initializing PPO Trainer...")
    try:
        runner = OnPolicyRunner(
            env,
            agent_cfg,
            log_dir,
            device=device
        )
        print(f"   ✓ Trainer initialized successfully")
    except Exception as e:
        print(f"   ✗ Failed to initialize trainer: {e}")
        return
    
    # Training parameters
    max_iterations = args.max_iterations if args.max_iterations else 5000
    
    print(f"\n📈 Training Configuration:")
    print(f"   Max Iterations: {max_iterations}")
    print(f"   Steps per Iteration: {args.num_envs * 20}  # num_envs × steps per env")
    print(f"   Total Steps: {max_iterations * args.num_envs * 20:,}")
    print(f"   Estimated Duration: ~{max_iterations * args.num_envs * 20 / 1000 / 50:.1f} minutes at 50 Hz sim")
    
    # Save configuration to log directory
    config_dict = {
        "num_envs": env_cfg.num_envs,
        "headless": args.headless,
        "max_iterations": max_iterations,
        "device": str(device),
        "timestamp": datetime.datetime.now().isoformat(),
    }
    with open(log_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    # Start training
    print(f"\n{'='*80}")
    print(f"🚀 STARTING TRAINING...")
    print(f"{'='*80}\n")
    
    try:
        # Train
        runner.learn(num_learning_iterations=max_iterations, init_at_random_ep_len=True)
        
        print(f"\n{'='*80}")
        print(f"✓ TRAINING COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        print(f"Logs saved to: {log_dir}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n✗ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print(f"\n🧹 Cleaning up...")
        try:
            env.close()
        except Exception as e:
            print(f"   Warning: Failed to close environment: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Go1 walking policy with PPO")
    parser.add_argument("--num_envs", type=int, default=512, help="Number of parallel environments")
    parser.add_argument("--max_iterations", type=int, default=5000, help="Maximum training iterations")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--headless", action="store_true", default=True, help="Run in headless mode (no GUI)")
    
    args = parser.parse_args()
    
    main(args)
