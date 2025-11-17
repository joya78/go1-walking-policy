# Go1 Walking Policy - Setup Complete! 🎉

## ✅ What's Been Done

Your Isaac Lab environment is now fully configured with a custom Go1 walking policy project!

### Installed Components
- ✅ Isaac Sim 5.1.0 (via pip)
- ✅ Isaac Lab core framework
- ✅ PyTorch 2.7.0 + CUDA 12.8
- ✅ RL Frameworks: RSL RL, SKRL, Stable Baselines3, RL Games
- ✅ GPU Support: 4× NVIDIA RTX A4500 (20GB VRAM each)

### Project Structure Created

```
/home/maxime/my_go1_project/
├── README.md                          # Comprehensive documentation
├── SETUP_COMPLETE.md                  # This file
├── info.sh                            # Project information script
├── train.sh                           # Quick training launcher
├── config/
│   ├── go1_walking_env_cfg.py        # Environment configuration (customizable copy)
│   └── go1_ppo_cfg.py                # PPO hyperparameters (customizable copy)
└── scripts/
    ├── train_go1_walking.py          # Main training script
    ├── play_go1_walking.py           # Policy testing/visualization script
    └── verify_setup.py               # Environment verification
```

## 🚀 Start Training NOW

### Option 1: Quick Start (Recommended)
```bash
cd /home/maxime/my_go1_project
./train.sh
```

This will:
- Train on rough terrain (most challenging)
- Use 4096 parallel environments
- Run for 1500 iterations (~30-45 minutes)
- Run in headless mode (faster, no GUI)

### Option 2: Custom Training
```bash
cd /home/maxime/my_go1_project

# Train with fewer environments (if GPU memory limited)
./train.sh --num_envs 2048

# Train longer for better results
./train.sh --max_iterations 3000

# Train with visualization (slower but you can watch)
./train.sh --gui

# Train on flat terrain (easier, faster convergence)
./train.sh --task Isaac-Velocity-Flat-Unitree-Go1-v0
```

### Option 3: Advanced (Manual)
```bash
cd /home/maxime/IsaacLab-main
conda activate env_isaaclab

bash isaaclab.sh -p /home/maxime/my_go1_project/scripts/train_go1_walking.py \
    --task Isaac-Velocity-Rough-Unitree-Go1-v0 \
    --num_envs 4096 \
    --max_iterations 1500 \
    --headless
```

## 📊 Monitor Training Progress

### TensorBoard
```bash
# In a new terminal
cd /home/maxime/IsaacLab-main
conda activate env_isaaclab
tensorboard --logdir logs/rsl_rl/
```

Then open in browser: http://localhost:6006

### Key Metrics to Watch
- **Episode Reward**: Should increase over time
- **Episode Length**: Target ~500-1000 steps
- **Linear Velocity Tracking**: Should approach 1.0
- **Policy Loss**: Should decrease and stabilize

## 🎮 Test Your Trained Policy

After training completes:

```bash
cd /home/maxime/IsaacLab-main

# Find your checkpoint
ls logs/rsl_rl/unitree_go1_rough/

# Play the policy (replace <timestamp> and <iteration>)
bash isaaclab.sh -p /home/maxime/my_go1_project/scripts/play_go1_walking.py \
    --checkpoint logs/rsl_rl/unitree_go1_rough/<timestamp>/model_<iteration>.pt \
    --num_envs 50
```

## 🔧 Customize Your Policy

### Modify Rewards (Most Common)

Edit: `my_go1_project/config/go1_walking_env_cfg.py`

```python
# Increase velocity tracking importance
self.rewards.track_lin_vel_xy_exp.weight = 2.0  # default: 1.5

# Reduce torque penalty (allow more aggressive movements)
self.rewards.dof_torques_l2.weight = -0.0001  # default: -0.0002

# Increase gait quality reward
self.rewards.feet_air_time.weight = 0.05  # default: 0.01
```

Then retrain with your modifications!

### Change Terrain Difficulty

```python
# Make boxes higher
self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.05, 0.2)

# Make rough terrain more challenging
self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.02, 0.1)
```

### Adjust Training Hyperparameters

Edit: `my_go1_project/config/go1_ppo_cfg.py`

```python
# Change learning rate
self.policy.learning_rate = 0.0005  # default: 0.001

# More training steps per iteration
self.num_steps_per_env = 48  # default: 24

# More epochs per update
self.num_epochs = 8  # default: 5
```

## 💡 Tips for Success

### GPU Memory Management
If you get out-of-memory errors:
- Reduce `--num_envs` (try 2048, 1024, or 512)
- Use headless mode (`--headless`)
- Close other GPU-intensive applications

### Training Time Estimates
On your 4× RTX A4500 setup:
- **500 iterations**: ~10-15 minutes
- **1500 iterations**: ~30-45 minutes
- **3000 iterations**: ~1-1.5 hours

### Convergence Tips
- Start with flat terrain to verify setup
- Move to rough terrain for final policy
- Monitor TensorBoard - reward should steadily increase
- If stuck, try adjusting learning rate or reward weights

## 📚 Next Steps

1. **Run First Training**: Start with default settings to familiarize yourself
2. **Analyze Results**: Use TensorBoard to understand what's working
3. **Iterate on Rewards**: Adjust weights based on observed behavior
4. **Test Different Terrains**: Compare flat vs rough terrain performance
5. **Sim-to-Real Transfer**: Deploy on physical Go1 robot (requires hardware)

## 🔍 Troubleshooting

### "No module named 'omni'" Error
This is expected when running Python scripts directly. Always use:
```bash
bash isaaclab.sh -p <script>.py
```

### Training Not Starting
1. Check GPU availability: `nvidia-smi`
2. Verify conda environment: `conda activate env_isaaclab`
3. Ensure you're in Isaac Lab directory: `cd /home/maxime/IsaacLab-main`

### Policy Not Learning
- Check reward scaling (should see positive values in TensorBoard)
- Verify action scaling isn't too small or large
- Ensure termination conditions aren't too strict

## 📖 Documentation

- **Project README**: `/home/maxime/my_go1_project/README.md`
- **Isaac Lab Docs**: https://isaac-sim.github.io/IsaacLab
- **RSL RL**: https://github.com/leggedrobotics/rsl_rl

## 🎯 Your Files vs Original Isaac Lab

**Your customizable copies** (safe to modify):
- `my_go1_project/config/go1_walking_env_cfg.py`
- `my_go1_project/config/go1_ppo_cfg.py`
- `my_go1_project/scripts/train_go1_walking.py`
- `my_go1_project/scripts/play_go1_walking.py`

**Original Isaac Lab files** (reference only, don't edit):
- `IsaacLab-main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go1/`
- All core Isaac Lab source code

## 🎉 You're Ready!

Everything is set up and ready to go. Just run:

```bash
cd /home/maxime/my_go1_project
./train.sh
```

Happy training! 🤖🚀
