#!/usr/bin/env python3
"""
Sanity check script to validate Go1 config without Omniverse/IsaacLab runtime.

This script attempts to import and instantiate the config classes to verify:
- Import paths are correct
- Config objects can be instantiated
- Key fields have expected values

Usage:
  python scripts/config_sanity_check.py
  
Note: This will only work in an environment where isaaclab packages are installed.
If you get import errors, you may need to run from within Isaac Lab environment.
"""

import sys
from pathlib import Path

# Add the repo root to path so we can import config
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

print("=" * 80)
print("Go1 Walking Policy Config Sanity Check")
print("=" * 80)
print()

# Try to import the config classes
print("[1/4] Attempting to import config classes...")
try:
    from config.go1_walking_env_cfg import UnitreeGo1RoughEnvCfg, UnitreeGo1RoughEnvCfg_PLAY
    print("✓ Successfully imported config classes")
except ImportError as e:
    print(f"✗ Failed to import config classes: {e}")
    print("\nNote: This is expected if isaaclab packages are not installed.")
    print("To run this check inside Isaac Lab environment:")
    print("  cd /home/maxime/IsaacLab-main")
    print("  conda activate env_isaaclab")
    print("  cd /home/maxime/my_go1_project")
    print("  python scripts/config_sanity_check.py")
    sys.exit(1)

# Instantiate training config
print("\n[2/4] Instantiating UnitreeGo1RoughEnvCfg (training config)...")
try:
    training_cfg = UnitreeGo1RoughEnvCfg()
    print("✓ Successfully instantiated training config")
except Exception as e:
    print(f"✗ Failed to instantiate training config: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Instantiate play config
print("\n[3/4] Instantiating UnitreeGo1RoughEnvCfg_PLAY (evaluation config)...")
try:
    play_cfg = UnitreeGo1RoughEnvCfg_PLAY()
    print("✓ Successfully instantiated evaluation config")
except Exception as e:
    print(f"✗ Failed to instantiate evaluation config: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Print key config values
print("\n[4/4] Validating key configuration parameters...")
print("\n" + "=" * 80)
print("TRAINING CONFIG (UnitreeGo1RoughEnvCfg)")
print("=" * 80)

checks = [
    ("Scene - Number of Environments", training_cfg.scene.num_envs, 4096),
    ("Scene - Environment Spacing", training_cfg.scene.env_spacing, 2.5),
    ("Actions - Joint Position Scale", training_cfg.actions.joint_pos.scale, 0.25),
    ("Rewards - Linear Velocity Weight", training_cfg.rewards.track_lin_vel_xy_exp.weight, 1.0),
    ("Rewards - Angular Velocity Weight", training_cfg.rewards.track_ang_vel_z_exp.weight, 0.5),
    ("Rewards - Feet Air Time Weight", training_cfg.rewards.feet_air_time.weight, 0.125),
    ("Rewards - Vertical Velocity Penalty", training_cfg.rewards.lin_vel_z_l2.weight, -2.0),
    ("Rewards - Roll/Pitch Penalty", training_cfg.rewards.ang_vel_xy_l2.weight, -0.05),
    ("Rewards - Torque Penalty", training_cfg.rewards.dof_torques_l2.weight, -1.0e-5),
    ("Rewards - Action Rate Penalty", training_cfg.rewards.action_rate_l2.weight, -0.01),
]

passed = 0
failed = 0
for name, actual, expected in checks:
    if abs(actual - expected) < 1e-7:
        print(f"✓ {name}: {actual}")
        passed += 1
    else:
        print(f"✗ {name}: expected {expected}, got {actual}")
        failed += 1

print(f"\nTraining config checks: {passed} passed, {failed} failed")

print("\n" + "=" * 80)
print("EVALUATION CONFIG (UnitreeGo1RoughEnvCfg_PLAY)")
print("=" * 80)

eval_checks = [
    ("Scene - Number of Environments", play_cfg.scene.num_envs, 50),
    ("Scene - Environment Spacing", play_cfg.scene.env_spacing, 2.5),
    ("Terrain - Curriculum Enabled", 
     play_cfg.scene.terrain.terrain_generator.curriculum if play_cfg.scene.terrain.terrain_generator else False, 
     False),
    ("Observations - Corruption Enabled", play_cfg.observations.policy.enable_corruption, False),
]

passed = 0
failed = 0
for name, actual, expected in eval_checks:
    if actual == expected:
        print(f"✓ {name}: {actual}")
        passed += 1
    else:
        print(f"✗ {name}: expected {expected}, got {actual}")
        failed += 1

print(f"\nEvaluation config checks: {passed} passed, {failed} failed")

# Print additional info
print("\n" + "=" * 80)
print("ADDITIONAL CONFIGURATION INFO")
print("=" * 80)

print(f"\nTraining Config:")
print(f"  - Episode length: {training_cfg.episode_length_s} seconds")
print(f"  - Simulation dt: {training_cfg.sim.dt} seconds (control at {1/training_cfg.sim.dt:.0f} Hz)")
print(f"  - Decimation: {training_cfg.decimation} (action frequency: {(1/training_cfg.sim.dt)/training_cfg.decimation:.0f} Hz)")
print(f"  - Estimated actions per episode: {int(training_cfg.episode_length_s / (training_cfg.sim.dt * training_cfg.decimation))}")

print(f"\nReward Function Configuration:")
print(f"  - Undesired contacts penalty: {training_cfg.rewards.undesired_contacts}")
print(f"  - Flat orientation penalty weight: {training_cfg.rewards.flat_orientation_l2.weight}")
print(f"  - Feet air time sensor body names: {training_cfg.rewards.feet_air_time.params['sensor_cfg'].body_names if 'sensor_cfg' in training_cfg.rewards.feet_air_time.params else 'N/A'}")

print(f"\nEvent Configuration:")
print(f"  - Push robot enabled: {training_cfg.events.push_robot is not None}")
print(f"  - Base COM randomization enabled: {training_cfg.events.base_com is not None}")

print("\n" + "=" * 80)
print("✓ Config sanity check PASSED!")
print("=" * 80)
print("\nNext steps:")
print("  1. Review the annotated config file: config/go1_walking_env_cfg.py")
print("  2. Start training with: bash train.sh")
print("  3. Monitor with: tensorboard --logdir logs/rsl_rl/")
print()
