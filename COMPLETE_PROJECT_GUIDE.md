# Go1 Walking Policy - Complete Project Guide

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture & Components](#architecture--components)
3. [Quick Start Guide](#quick-start-guide)
4. [Configuration System](#configuration-system)
5. [Training & Evaluation](#training--evaluation)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Topics](#advanced-topics)

---

## Project Overview

### What is This?

This is a **self-contained reinforcement learning project** for training a Go1 robot locomotion policy using:
- **Isaac Lab**: Physics simulation framework (NVIDIA Omniverse-based)
- **RSL-RL**: On-policy PPO reinforcement learning algorithm
- **PyTorch**: Deep learning framework for neural networks
- **4096 parallel environments**: For data-efficient training

### Goal

Train a neural network policy that maps robot observations (position, velocity, IMU) → joint angle commands, enabling the Go1 robot to walk at commanded velocities across rough terrain.

### Key Statistics

- **Training Time**: ~30-60 minutes (1500 iterations on RTX A4500)
- **Parallel Environments**: 4096 during training, 50 during evaluation
- **Total Transitions**: ~147 million transitions per training run
- **Model Size**: ~100KB (small enough for robot deployment)
- **GPU Memory**: ~20GB (fits on single RTX A4500)

---

## Architecture & Components

### File Structure

```
go1-walking-policy/
├── config/                          # Configuration package
│   ├── __init__.py                  # Module initialization
│   ├── go1_walking_env_cfg.py       # Go1-specific environment config
│   ├── agents/
│   │   └── rsl_rl_ppo_cfg.py        # PPO hyperparameters
│   ├── base/
│   │   └── velocity_env_cfg.py      # Base locomotion environment
│   ├── mdp/
│   │   ├── __init__.py
│   │   ├── actions.py               # Action space definitions
│   │   ├── observations.py          # Observation/state definitions
│   │   ├── rewards.py               # Reward function implementations
│   │   ├── terminations.py          # Episode termination conditions
│   │   ├── events.py                # Env events (resets, disturbances)
│   │   └── curriculums.py           # Curriculum learning strategies
│   └── __pycache__/
├── scripts/                         # Utility and training scripts
│   ├── train_go1_walking.py         # Main training orchestrator
│   ├── play_go1_walking.py          # Policy evaluation/playback
│   ├── train_custom.py              # Custom training loop
│   ├── inspect_checkpoint.py        # Checkpoint inspection
│   ├── analyze_training.py          # Training metrics analysis
│   ├── visualize_*.py               # Visualization scripts
│   └── [others]
├── train.sh                         # Training launcher (shell wrapper)
├── test.sh                          # Evaluation launcher (shell wrapper)
├── README.md                        # Project documentation
├── SETUP_COMPLETE.md                # Setup verification
├── VERIFICATION.md                  # Verification steps
├── PROJECT_STRUCTURE.md             # Detailed file descriptions
└── videos/                          # Recorded evaluation videos
```

### Core Components

#### 1. **Environment Configuration** (`config/`)

**`go1_walking_env_cfg.py`** (185 lines)
- Customizes the base locomotion environment for Go1 robot
- Specifies terrain characteristics (rough with 1-6cm noise)
- Defines reward weights and termination conditions
- Configures action scaling and observation space
- Includes both training and evaluation variants

**`base/velocity_env_cfg.py`** (335 lines)
- Base locomotion environment using ManagerBasedRLEnv
- Configures MDP components:
  - **Scene**: Terrain, height scanner, contact sensors
  - **Commands**: Velocity targets (lin_x/y ±1.0 m/s, ang_z ±1.0 rad/s)
  - **Actions**: Joint position control (10 DOF robot)
  - **Observations**: State vector (24 dims) including velocity, gravity, joint state
  - **Events**: Random resets, mass perturbations, push disturbances
  - **Rewards**: 10+ terms (velocity tracking, efficiency, stability)
  - **Curriculum**: Terrain difficulty progression based on performance

**`agents/rsl_rl_ppo_cfg.py`** (50 lines)
- PPO hyperparameters: learning rate, network architecture, rollout length
- Experiment name and logging configuration
- 1500 training iterations, 24 steps per environment

**`mdp/` Package**
- **`rewards.py`**: Reward function implementations (feet air time, velocity tracking, penalties)
- **`terminations.py`**: Termination conditions (trunk contact = fall)
- **`observations.py`**: State vector components
- **`actions.py`**: Action space (joint angle targets)
- **`curriculums.py`**: Terrain difficulty progression

#### 2. **Training Pipeline** (`scripts/train_go1_walking.py`)

```
Step 1: Load Configs
  ↓ Parse environment config (Isaac-Velocity-Rough-Unitree-Go1-v0)
  ↓ Parse agent config (PPO hyperparameters)
  ↓ Override with command-line args

Step 2: Create Environment
  ↓ Instantiate 4096 parallel Go1 robots in physics simulation
  ↓ Initialize terrain, sensors, action/observation spaces

Step 3: Create Agent
  ↓ Build actor (policy) network: obs → actions
  ↓ Build critic (value) network: obs → value estimate
  ↓ Initialize optimizer (Adam)

Step 4: Training Loop (1500 iterations)
  ├─ Collect 4096 envs × 24 steps = 98,304 transitions
  ├─ Compute advantages using critic estimate
  ├─ Update actor (policy gradient)
  ├─ Update critic (value function)
  ├─ Log metrics (reward, loss, entropy)
  └─ Save checkpoint every 100 steps

Step 5: Cleanup
  ↓ Close environments and Isaac Lab
  ↓ Save final model
```

#### 3. **Evaluation Pipeline** (`scripts/play_go1_walking.py`)

```
Step 1: Load Checkpoint
  ↓ Extract trained actor network weights
  ↓ Load into eval-mode (no gradients)

Step 2: Create Evaluation Environment
  ↓ Instantiate 50 parallel Go1 robots
  ↓ Use -Play-v0 variant (deterministic, no curriculum)

Step 3: Rollout Loop
  ├─ Get obs from environment
  ├─ Forward through policy: obs → actions
  ├─ Step environment with actions
  ├─ [Optional] Record video frames
  ├─ Track episode statistics
  └─ Continue until stopping

Step 4: Output
  ↓ Print episode rewards/lengths
  ↓ Save video (if --video enabled)
```

---

## Quick Start Guide

### Prerequisites

✅ Already completed:
- Isaac Lab installed at `/home/ethan/IsaacLab`
- Conda environment `isaac_lab` created
- Project paths fixed in `train.sh` and `test.sh`
- CUDA 13.0 and RTX A4500 GPUs available

### 1. Verify Installation

```bash
source /home/ethan/miniconda3/bin/activate isaac_lab
cd /home/ethan/IsaacLab
python -c "from isaaclab.app import AppLauncher; print('✅ Isaac Lab OK')"
```

### 2. Train the Policy

```bash
cd /home/ethan/go1-walking-policy

# Basic training (4096 envs, 1500 iterations, headless)
bash train.sh

# Or with options:
bash train.sh --num_envs 4096 --max_iterations 1500 --headless

# Training with fewer envs (for testing/debugging)
bash train.sh --num_envs 512

# Resume from last checkpoint
bash train.sh --resume

# Training with GUI (slower, useful for debugging)
bash train.sh --gui
```

**Expected output:**
```
[INFO] Starting training for 1500 iterations...
[INFO] Iteration 1/1500 | Avg Reward: 0.123 | Loss: 0.456
[INFO] Iteration 100/1500 | Avg Reward: 2.345 | Loss: 0.234
...
[INFO] Training completed! Checkpoints saved to: logs/rsl_rl/unitree_go1_rough/2025-11-16_XX-XX-XX/
```

**Training checkpoints saved to:**
```
logs/rsl_rl/unitree_go1_rough/
├── 2025-11-16_10-30-00/
│   ├── model_0.pt              # Initial checkpoint
│   ├── model_100.pt
│   ├── model_500.pt
│   ├── model_1000.pt
│   ├── model_1500.pt           # Final checkpoint
│   ├── logs.csv                # Training metrics
│   └── config.yaml             # Saved config
```

### 3. Test the Trained Policy

```bash
# Test with latest checkpoint (auto-detected)
bash test.sh

# Test specific checkpoint
bash test.sh --checkpoint logs/rsl_rl/unitree_go1_rough/2025-11-16_10-30-00/model_1500.pt

# Test with video recording
bash test.sh --video

# Test with fewer environments
bash test.sh --num_envs 10

# Test with GUI
bash test.sh --gui
```

**Expected output:**
```
[INFO] Loading checkpoint from: logs/rsl_rl/unitree_go1_rough/2025-11-16_10-30-00/model_1500.pt
[INFO] Starting policy playback...
[INFO] Episode 1 | Steps: 500 | Avg Reward: 2.123
[INFO] Episode 2 | Steps: 480 | Avg Reward: 2.156
...
```

**Recorded videos saved to:**
```
videos/go1_policy_20251116_HHMMSS.mp4
```

---

## Configuration System

### How Configurations Work

The project uses a hierarchical config system:

```
go1_walking_env_cfg.LocomotionVelocityRoughEnvCfg  (Go1 specific)
    ↑
    └─ inherits from velocity_env_cfg.LocomotionVelocityRoughEnvCfg (base)
        ↑
        └─ uses mdp.* components (rewards, actions, observations, etc.)

go1_walking_env_cfg.RslRlPPOCfgGo1Rough            (PPO hyperparameters)
    ↑
    └─ rsl_rl_ppo_cfg.RslRlOnPolicyRunnerCfg        (base config)
```

### Key Configuration Parameters

| Parameter | Location | Default | Meaning |
|-----------|----------|---------|---------|
| `num_envs` | `scene.num_envs` | 4096 | Parallel environments (training) |
| `terrain_type` | `scene.terrain.terrain_type` | "generator" | Procedurally generated rough |
| `action_scale` | `actions.joint_pos.scale` | 0.25 | Joint angle magnitude limit |
| `command_range` | `commands.*` | ±1.0 m/s | Linear/angular velocity targets |
| `reward_weights` | `rewards.*` | varies | Per-term reward scaling |
| `max_iterations` | agent config | 1500 | Training steps |
| `learning_rate` | agent config | 1e-3 | Optimizer learning rate |

### Customizing Configuration

**Example: Train on flat terrain with fewer environments**

```python
# scripts/train_custom.py
from isaaclab_tasks.utils import parse_env_cfg
import gymnasium as gym

# Load flat terrain config instead of rough
env_cfg = parse_env_cfg(
    "Isaac-Velocity-Flat-Unitree-Go1-v0",  # Flat instead of Rough
    device="cuda:0",
    num_envs=1024  # Fewer environments
)

# Adjust rewards to prefer higher speeds
env_cfg.rewards.lin_vel_xy_exp.weight = 2.0  # Increase from 1.0

# Create and train
env = gym.make("Isaac-Velocity-Flat-Unitree-Go1-v0", cfg=env_cfg)
# ... training code ...
```

---

## Training & Evaluation

### Training Metrics

The training script logs these metrics to `logs.csv`:

| Metric | Meaning |
|--------|---------|
| `rewards/episode_reward` | Average episode return |
| `rewards/total_reward` | Cumulative reward during rollout |
| `rewards/lin_vel` | Linear velocity reward component |
| `rewards/ang_vel` | Angular velocity reward component |
| `policy_loss` | Actor network loss |
| `value_loss` | Critic network loss |
| `policy_entropy` | Policy exploration (higher = more exploration) |

**View training progress:**
```bash
cd /home/ethan/go1-walking-policy
python scripts/analyze_training.py logs/rsl_rl/unitree_go1_rough/2025-11-16_XX-XX-XX/logs.csv
```

### Evaluation Metrics

During playback, the script reports:

| Metric | Meaning |
|--------|---------|
| `Episode` | Episode number |
| `Steps` | How many steps before episode ended |
| `Avg Reward` | Average reward per step |

**Good performance:** Avg Reward > 2.0, Episode length > 400 steps

### Checkpoint Management

**Save checkpoints:**
- Automatically saved every 100 training iterations
- Located in `logs/rsl_rl/unitree_go1_rough/TIMESTAMP/`
- Latest: `model_1500.pt` (final trained model)

**Resume training:**
```bash
bash train.sh --resume              # Auto-detect latest
bash train.sh --checkpoint PATH     # Use specific checkpoint
```

**Compare checkpoints:**
```bash
bash test.sh --checkpoint logs/rsl_rl/unitree_go1_rough/2025-11-16_XX-XX-XX/model_500.pt
bash test.sh --checkpoint logs/rsl_rl/unitree_go1_rough/2025-11-16_XX-XX-XX/model_1500.pt
# Compare reward curves to see improvement
```

---

## Troubleshooting

### Issue: "Isaac Lab not found at /home/ethan/IsaacLab"

**Solution:**
```bash
# Verify installation
ls -la /home/ethan/IsaacLab/isaaclab.sh

# If missing, reinstall:
bash /home/ethan/install_isaac_lab_pip.sh
```

### Issue: "ModuleNotFoundError: No module named 'isaaclab'"

**Solution:**
```bash
# Activate correct environment
source /home/ethan/miniconda3/bin/activate isaac_lab

# Verify Isaac Lab is installed
python -c "import isaaclab; print(isaaclab.__version__)"

# If not installed, reinstall in environment
pip install -e /home/ethan/IsaacLab
```

### Issue: "Out of memory" during training

**Symptoms:** Training crashes with CUDA OOM error after 30-60 seconds

**Solutions:**
```bash
# Option 1: Reduce environments
bash train.sh --num_envs 2048

# Option 2: Use smaller physics timestep
# (edit go1_walking_env_cfg.py, reduce sim.dt)

# Option 3: Check GPU memory usage
nvidia-smi

# Option 4: Close other GPU processes
pkill -f "python.*play_go1"
```

### Issue: Training very slow

**Possible causes:**
1. GUI enabled (`--gui` flag)
2. Too many environments for GPU (reduce with `--num_envs`)
3. Disk I/O bottleneck (disable video recording)

**Check performance:**
```bash
nvidia-smi dmon  # Real-time GPU utilization
htop             # CPU and memory usage
```

### Issue: "No checkpoints found" when running test.sh

**Cause:** No training has run yet, or checkpoint in wrong location

**Solution:**
```bash
# First, complete at least 1 training iteration
bash train.sh --max_iterations 1

# Wait for checkpoint to save (~10-20 seconds)

# Then run testing
bash test.sh
```

### Issue: Video recording produces corrupted file

**Solution:**
```bash
# Install FFmpeg
sudo apt-get install ffmpeg

# Re-install imageio with FFmpeg
pip install --upgrade imageio imageio-ffmpeg

# Re-run with video
bash test.sh --video
```

---

## Advanced Topics

### Multi-GPU Training

Isaac Lab supports distributed training using PyTorch Distributed:

```bash
# Automatically detects and uses all GPUs
bash train.sh --num_envs 8192  # 2× normal (uses both GPUs)
```

### Custom Reward Functions

Edit `config/mdp/rewards.py` to add new reward terms:

```python
def custom_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Example: penalize high joint velocities for smooth motion."""
    joint_vel = env.unwrapped.data.joint_vel[:, 6:16]  # Leg joint velocities
    return -torch.norm(joint_vel, dim=1) * 0.01  # Small penalty

# Then add to config:
env_cfg.rewards.custom = RewardTermCfg(
    func=mdp.custom_reward,
    weight=1.0
)
```

### Training with Different Terrain

```bash
# Flat terrain (easier)
bash train.sh --task Isaac-Velocity-Flat-Unitree-Go1-v0

# Rough terrain (default, harder)
bash train.sh --task Isaac-Velocity-Rough-Unitree-Go1-v0

# Other terrains (if registered)
bash train.sh --task Isaac-Velocity-[OTHER]-Unitree-Go1-v0
```

### Curriculum Learning

The environment includes curriculum learning that progressively increases terrain difficulty based on the agent's performance:

- **Phase 0** (iterations 0-100): Flat terrain, agent learns basic locomotion
- **Phase 1** (iterations 100-500): Low roughness (1-2cm), agent adapts to slight obstacles
- **Phase 2** (iterations 500-1500): Full roughness (1-6cm), agent learns robust walking

Curriculum is defined in `config/mdp/curriculums.py`:

```python
def terrain_levels_vel(env, env_cfg):
    """Returns terrain difficulty based on walking distance."""
    if walking_distance < 5.0:
        return 0  # Flat
    elif walking_distance < 15.0:
        return 1  # Light roughness
    else:
        return 2  # Full roughness
```

### Monitoring with TensorBoard

```bash
# View training metrics during training
cd /home/ethan/go1-walking-policy
tensorboard --logdir=logs/rsl_rl/ --port=6006

# Open browser to http://localhost:6006
```

### Exporting for Robot Deployment

After training, export the checkpoint for onboard robot execution:

```bash
python scripts/inspect_checkpoint.py logs/rsl_rl/unitree_go1_rough/TIMESTAMP/model_1500.pt

# Output: Neural network architecture, weight shapes, inference specs
# Can then be loaded on robot using:
# - TorchScript (torch.jit.load)
# - ONNX (torch.onnx.export)
# - Native PyTorch (torch.load)
```

---

## Next Steps

1. **Verify Installation**
   ```bash
   source /home/ethan/miniconda3/bin/activate isaac_lab
   cd /home/ethan/IsaacLab
   python -c "from isaaclab.app import AppLauncher; print('✅ OK')"
   ```

2. **Run Training**
   ```bash
   cd /home/ethan/go1-walking-policy
   bash train.sh --num_envs 512  # Start with fewer envs for testing
   ```

3. **Monitor Progress**
   ```bash
   # In another terminal:
   watch nvidia-smi  # Monitor GPU usage
   ```

4. **Evaluate Results**
   ```bash
   bash test.sh --video  # Test with video recording
   ```

5. **Deploy to Robot**
   - Export checkpoint using `inspect_checkpoint.py`
   - Load on robot's onboard compute
   - Stream observations and execute actions

---

## References

- **Isaac Lab Docs**: https://isaac-sim.github.io/IsaacLab/
- **RSL-RL GitHub**: https://github.com/leggedrobotics/rsl_rl
- **Unitree Go1**: https://www.unitree.com/en/go1
- **PPO Algorithm**: https://arxiv.org/abs/1707.06347

