#!/usr/bin/env python3
"""Verify Isaac Lab and Go1 environment setup."""

import sys

def check_imports():
    """Check if all required modules can be imported."""
    print("=" * 60)
    print("Checking Python environment and imports...")
    print("=" * 60)
    
    errors = []
    
    # Check Python version
    print(f"\n✓ Python version: {sys.version}")
    
    # Check Isaac Lab
    try:
        import isaaclab
        print(f"✓ Isaac Lab installed")
    except ImportError as e:
        errors.append(f"✗ Isaac Lab not found: {e}")
    
    # Check Isaac Lab tasks
    try:
        import isaaclab_tasks
        print(f"✓ Isaac Lab Tasks installed")
    except ImportError as e:
        errors.append(f"✗ Isaac Lab Tasks not found: {e}")
    
    # Check PyTorch
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} installed")
        print(f"  - CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  - CUDA version: {torch.version.cuda}")
            print(f"  - GPU devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    - GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError as e:
        errors.append(f"✗ PyTorch not found: {e}")
    
    # Check RL libraries
    try:
        import rsl_rl
        print(f"✓ RSL RL installed")
    except ImportError as e:
        errors.append(f"✗ RSL RL not found: {e}")
    
    try:
        import skrl
        print(f"✓ SKRL installed")
    except ImportError as e:
        print(f"⚠ SKRL not found (optional): {e}")
    
    try:
        import stable_baselines3
        print(f"✓ Stable Baselines3 installed")
    except ImportError as e:
        print(f"⚠ Stable Baselines3 not found (optional): {e}")
    
    # Check gymnasium
    try:
        import gymnasium as gym
        print(f"✓ Gymnasium installed")
    except ImportError as e:
        errors.append(f"✗ Gymnasium not found: {e}")
    
    if errors:
        print("\n" + "=" * 60)
        print("ERRORS FOUND:")
        print("=" * 60)
        for error in errors:
            print(error)
        return False
    else:
        print("\n" + "=" * 60)
        print("✓ All required packages are installed!")
        print("=" * 60)
        return True


def check_environments():
    """Check if Go1 environments are registered."""
    print("\n" + "=" * 60)
    print("Checking Go1 environments...")
    print("=" * 60)
    
    try:
        import gymnasium as gym
        
        go1_envs = [
            "Isaac-Velocity-Rough-Unitree-Go1-v0",
            "Isaac-Velocity-Rough-Unitree-Go1-Play-v0",
            "Isaac-Velocity-Flat-Unitree-Go1-v0",
            "Isaac-Velocity-Flat-Unitree-Go1-Play-v0",
        ]
        
        registered_envs = [env_id for env_id in gym.envs.registry.keys()]
        
        found_envs = []
        missing_envs = []
        
        for env_id in go1_envs:
            if env_id in registered_envs:
                found_envs.append(env_id)
                print(f"✓ {env_id}")
            else:
                missing_envs.append(env_id)
                print(f"✗ {env_id} - NOT FOUND")
        
        print(f"\nFound {len(found_envs)}/{len(go1_envs)} Go1 environments")
        
        if missing_envs:
            print("\nMissing environments:")
            for env_id in missing_envs:
                print(f"  - {env_id}")
            return False
        else:
            print("\n✓ All Go1 environments are registered!")
            return True
            
    except Exception as e:
        print(f"\n✗ Error checking environments: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_system_info():
    """Print system information."""
    print("\n" + "=" * 60)
    print("System Information")
    print("=" * 60)
    
    import os
    import platform
    
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"CPU: {platform.processor()}")
    print(f"Python: {platform.python_version()}")
    print(f"Working Directory: {os.getcwd()}")


def main():
    """Run all verification checks."""
    print("\n" + "=" * 60)
    print("Isaac Lab Go1 Environment Verification")
    print("=" * 60)
    
    print_system_info()
    
    # Check imports
    imports_ok = check_imports()
    
    # Check environments (only if imports are OK)
    if imports_ok:
        envs_ok = check_environments()
        
        if envs_ok:
            print("\n" + "=" * 60)
            print("✓ Setup verification PASSED!")
            print("=" * 60)
            print("\nYou're ready to train the Go1 walking policy!")
            print("\nNext steps:")
            print("  1. cd /home/maxime/IsaacLab-main")
            print("  2. bash isaaclab.sh -p my_go1_project/scripts/train_go1_walking.py --headless")
            return 0
        else:
            print("\n" + "=" * 60)
            print("✗ Environment check FAILED")
            print("=" * 60)
            return 1
    else:
        print("\n" + "=" * 60)
        print("✗ Import check FAILED")
        print("=" * 60)
        print("\nPlease ensure Isaac Lab is properly installed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
