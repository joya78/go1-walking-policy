# Go1 Walking Policy Project

This project contains a **complete, self-contained** configuration for training and testing a low-level walking policy on the Unitree Go1 robot using Isaac Lab and RSL-RL.

## 📦 What's Included

All necessary files from IsaacLab are now **included in this repository**:
- ✅ Base locomotion environment configuration (`config/base/`)
- ✅ MDP definitions: rewards, terminations, curriculum (`config/mdp/`)
- ✅ Go1-specific configuration (`config/go1_walking_env_cfg.py`)
- ✅ PPO training configuration (`config/agents/rsl_rl_ppo_cfg.py`)
- ✅ Training and testing scripts (`scripts/`)

See [FILES_INCLUDED.md](FILES_INCLUDED.md) for complete details.

## Project Structure

```
my_go1_project/
├── README.md                          # This file
├── FILES_INCLUDED.md                  # Documentation of included files
├── PROJECT_STRUCTURE.md               # Detailed structure explanation
├── config/
│   ├── __init__.py                   # Gym environment registration
│   ├── go1_walking_env_cfg.py        # Go1-specific configuration
│   ├── agents/
│   │   └── rsl_rl_ppo_cfg.py        # PPO hyperparameters
│   ├── base/
│   │   └── velocity_env_cfg.py       # Base locomotion config (from IsaacLab)
│   └── mdp/
│       ├── rewards.py                # Reward functions (from IsaacLab)
│       ├── terminations.py           # Termination conditions (from IsaacLab)
│       └── curriculums.py            # Curriculum learning (from IsaacLab)
├── scripts/
│   ├── train_go1_walking.py         # Training script
│   └── play_go1_walking.py          # Testing/evaluation script
├── train.sh                          # Quick training shortcut
└── test.sh                           # Quick testing shortcut
```

## Environment Setup

The project uses the Isaac Lab conda environment:
```bash
conda activate env_isaaclab
cd /home/maxime/IsaacLab-main
```

## Available Go1 Environments

## 🚀 Quick Start

### Training
```bash
bash train.sh
```

Or with custom parameters:
```bash
cd /home/maxime/IsaacLab-main
bash isaaclab.sh -p /home/maxime/my_go1_project/scripts/train_go1_walking.py \
    --task Isaac-Velocity-Rough-Unitree-Go1-v0 \
    --num_envs 4096 \
    --max_iterations 1500 \
    --headless
```

### Testing
```bash
bash test.sh --checkpoint logs/rsl_rl/unitree_go1_rough/2025-11-16_XX-XX-XX/model_XXX.pt
```

With video recording:
```bash
bash test.sh --checkpoint MODEL.pt --video
```

## 🎮 Available Environments

This project provides **custom Go1 environments** with all configurations included:

1. **Isaac-Velocity-Rough-Unitree-Go1-Custom-v0** - Training on rough terrain
2. **Isaac-Velocity-Rough-Unitree-Go1-Custom-Play-v0** - Testing/evaluation

These are self-contained versions that use the local configuration files in this repo.

## ⚙️ Configuration Details

### Environment (`config/go1_walking_env_cfg.py`)

**Temporal settings:**
- Episode length: 20 seconds
- Simulation dt: 0.005s (200 Hz)
- Action frequency: 50 Hz (decimation=4)
- ~1000 actions per episode

**Key features:**
- Terrain adapted for Go1 size
- Action scale: 0.25 for smooth movements
- Mass randomization: -1kg to +3kg
- Contact monitoring on feet (".*_foot") and trunk

### Rewards (configurable in `config/go1_walking_env_cfg.py`)

- ✅ `track_lin_vel_xy_exp` (weight: 1.0) - Track linear velocity
- ✅ `track_ang_vel_z_exp` (weight: 0.5) - Track angular velocity
- ✅ `feet_air_time` (weight: 0.125) - Natural gait
- ⚠️ `lin_vel_z_l2` (weight: -2.0) - Penalize vertical motion
- ⚠️ `ang_vel_xy_l2` (weight: -0.05) - Penalize roll/pitch
- ⚠️ `dof_torques_l2` (weight: -1e-5) - Penalize high torques
- ⚠️ `action_rate_l2` (weight: -0.01) - Smooth actions
- ⚠️ `flat_orientation_l2` (weight: -5.0) - Stay upright

**To modify:** Edit weights in `config/go1_walking_env_cfg.py`, section `self.rewards.*`

### PPO Configuration (`config/agents/rsl_rl_ppo_cfg.py`)

Training hyperparameters:
- **Max iterations**: 1500 (rough) / 300 (flat)
- **Steps per env**: 24 (per iteration)
- **Learning rate**: 0.001 (adaptive schedule)
- **Mini batches**: 4
- **Epochs**: 5
- **Network**: Actor/Critic with [512, 256, 128] hidden layers
- **Discount factor (gamma)**: 0.99
- **GAE lambda**: 0.95

**Total training steps**: 1500 iterations × 24 steps × 4096 envs = ~147M steps

## 📊 Monitoring Training

Training logs and checkpoints are saved to:
```
logs/rsl_rl/unitree_go1_rough/<timestamp>/
```

Monitor with TensorBoard:
```bash
tensorboard --logdir logs/rsl_rl/
```

## 🎯 Understanding Episodes vs Iterations

### Episode (Testing)
- **Duration**: 20 seconds
- **Steps**: ~1000 actions (50 Hz)
- **Reset**: When robot falls or time expires
- **Purpose**: Evaluate policy performance

### Iteration (Training)
- **Duration**: 24 steps per environment (0.48s)
- **Data collected**: 24 × 4096 = 98,304 transitions
- **Updates**: 5 epochs × 4 mini-batches
- **Purpose**: Collect data and update policy

**Training progress**: 1 robot completes ~42 episodes per 1000 iterations

## 🛠️ Customization Guide

### 1. Modify Reward Weights

Edit `config/go1_walking_env_cfg.py`:
```python
# Increase velocity tracking
self.rewards.track_lin_vel_xy_exp.weight = 2.0

# Penalize energy more
self.rewards.dof_torques_l2.weight = -0.0001

# Add/remove rewards
self.rewards.undesired_contacts = None  # Disable
```

### 2. Add New Reward Functions

Edit `config/mdp/rewards.py` to add custom rewards, then use in `go1_walking_env_cfg.py`.

### 3. Change Terrain

```python
# In go1_walking_env_cfg.py
self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.05, 0.15)
```

### 4. Adjust Training Speed

```python
# In config/agents/rsl_rl_ppo_cfg.py
self.num_steps_per_env = 48  # Collect more steps per iteration
self.max_iterations = 3000    # Train longer
```

## 💾 Checkpoints and Resume

Resume training from checkpoint:
```bash
bash train.sh --resume  # Auto-finds latest
bash train.sh --checkpoint logs/.../model_500.pt  # Specific checkpoint
```

## ⚡ GPU Memory Optimization

If OOM (out of memory):
1. Reduce `--num_envs` (try 2048 or 1024)
2. Use `--headless` mode
3. Reduce terrain size in config
4. Use smaller network (`[256, 128, 64]`)

## 📚 Additional Documentation

- [FILES_INCLUDED.md](FILES_INCLUDED.md) - Complete list of included files
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - Detailed structure and concepts
- [SETUP_COMPLETE.md](SETUP_COMPLETE.md) - Setup verification

## Expected Training Time

On 4× RTX A4500 GPUs:
- 1000 iterations: ~15-30 minutes
- 3000 iterations: ~45-90 minutes (depends on terrain complexity)

## Next Steps

1. **Train baseline**: Run training with default settings
2. **Analyze performance**: Use TensorBoard to monitor rewards
3. **Tune rewards**: Adjust weights based on observed behavior
4. **Test sim-to-real**: Deploy policy on real Go1 robot (requires additional hardware setup)
