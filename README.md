# Go1 Walking Policy Project

This project contains custom configurations for training a low-level walking policy on the Unitree Go1 robot using Isaac Lab.

## Project Structure

```
my_go1_project/
├── README.md                          # This file
├── config/
│   ├── go1_walking_env_cfg.py        # Environment configuration (copied from Isaac Lab)
│   └── go1_ppo_cfg.py                # PPO training configuration (copied from Isaac Lab)
└── scripts/
    └── train_go1_walking.py          # Training script
```

## Environment Setup

The project uses the Isaac Lab conda environment:
```bash
conda activate env_isaaclab
cd /home/maxime/IsaacLab-main
```

## Available Go1 Environments

Isaac Lab provides several pre-configured Go1 environments:

1. **Isaac-Velocity-Rough-Unitree-Go1-v0** - Training on rough terrain
2. **Isaac-Velocity-Rough-Unitree-Go1-Play-v0** - Testing on rough terrain (fewer envs)
3. **Isaac-Velocity-Flat-Unitree-Go1-v0** - Training on flat terrain
4. **Isaac-Velocity-Flat-Unitree-Go1-Play-v0** - Testing on flat terrain (fewer envs)

## Training

### Quick Start (Using Default Configs)

Train on rough terrain with default settings:
```bash
bash isaaclab.sh -p my_go1_project/scripts/train_go1_walking.py --task Isaac-Velocity-Rough-Unitree-Go1-v0
```

### With Custom Parameters

```bash
bash isaaclab.sh -p my_go1_project/scripts/train_go1_walking.py \
    --task Isaac-Velocity-Rough-Unitree-Go1-v0 \
    --num_envs 4096 \
    --max_iterations 3000 \
    --headless
```

### Training Parameters

- `--task`: Environment ID (see list above)
- `--num_envs`: Number of parallel environments (default: 4096, use less if GPU memory limited)
- `--max_iterations`: Training iterations (default: 1500)
- `--headless`: Run without visualization (faster training)
- `--seed`: Random seed for reproducibility
- `--cpu`: Force CPU-only mode (not recommended)

## Configuration Files

### Environment Configuration (`config/go1_walking_env_cfg.py`)

Key parameters you can modify:
- **Observations**: Joint positions, velocities, base orientation, commands
- **Actions**: Joint position targets with PD control
- **Rewards**: 
  - `track_lin_vel_xy_exp`: Reward for tracking linear velocity commands (weight: 1.5)
  - `track_ang_vel_z_exp`: Reward for tracking angular velocity commands (weight: 0.75)
  - `dof_torques_l2`: Penalize high torques (weight: -0.0002)
  - `feet_air_time`: Reward for proper gait (weight: 0.01)
  - `dof_acc_l2`: Penalize high accelerations (weight: -2.5e-7)
- **Terrain**: Boxes, random rough surfaces, stairs, etc.
- **Terminations**: Base contact with ground, joint limits

### PPO Configuration (`config/go1_ppo_cfg.py`)

Training hyperparameters:
- Learning rate: 0.001
- Number of steps per update: 24
- Mini batches: 4
- Epochs: 5
- Discount factor (gamma): 0.99
- GAE lambda: 0.95

## Monitoring Training

Training logs and checkpoints are saved to:
```
logs/rsl_rl/unitree_go1_rough/<timestamp>/
```

You can monitor training with TensorBoard:
```bash
tensorboard --logdir logs/rsl_rl/
```

## Testing Trained Policy

After training, test the policy:
```bash
bash isaaclab.sh -p my_go1_project/scripts/play_go1_walking.py \
    --task Isaac-Velocity-Rough-Unitree-Go1-Play-v0 \
    --num_envs 50 \
    --checkpoint logs/rsl_rl/unitree_go1_rough/<timestamp>/model_<iteration>.pt
```

## Customization Guide

### Modifying Rewards

Edit `config/go1_walking_env_cfg.py` to adjust reward weights:
```python
self.rewards.track_lin_vel_xy_exp.weight = 2.0  # Increase velocity tracking reward
self.rewards.dof_torques_l2.weight = -0.001     # Penalize torques more
```

### Changing Terrain Difficulty

Adjust terrain parameters:
```python
self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.05, 0.15)
self.scene.terrain.terrain_generator.difficulty_range = (0.0, 1.0)
```

### Modifying Action Space

Change action scaling or control mode:
```python
self.actions.joint_pos.scale = 0.5  # Larger position deltas
```

## GPU Memory Optimization

If running out of GPU memory:
1. Reduce `--num_envs` (e.g., 2048 or 1024)
2. Use smaller terrain size
3. Enable headless mode with `--headless`

## Expected Training Time

On 4× RTX A4500 GPUs:
- 1000 iterations: ~15-30 minutes
- 3000 iterations: ~45-90 minutes (depends on terrain complexity)

## Next Steps

1. **Train baseline**: Run training with default settings
2. **Analyze performance**: Use TensorBoard to monitor rewards
3. **Tune rewards**: Adjust weights based on observed behavior
4. **Test sim-to-real**: Deploy policy on real Go1 robot (requires additional hardware setup)
