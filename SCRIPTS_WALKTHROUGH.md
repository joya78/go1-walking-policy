# Go1 Walking Policy - Scripts Walkthrough

## Overview
This document provides a line-by-line explanation of the key training and evaluation scripts in the Go1 walking policy project.

---

## 1. Shell Scripts (train.sh & test.sh)

### `train.sh` - Training Wrapper Script

**Purpose:** High-level shell wrapper to easily launch Isaac Lab training with command-line argument parsing.

#### Structure:

```bash
#!/bin/bash
# Quick start script for training Go1 walking policy
```
- Bash shebang; the script uses shell functions and variables extensively.

#### Color Output Setup:
```bash
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color
```
- ANSI color codes for terminal output (blue headers, green success, yellow warnings).
- `NC` (No Color) resets color after each printed line.

#### Configuration Detection:
```bash
ISAACLAB_DIR="/home/maxime/IsaacLab-main"
if [ ! -f "$ISAACLAB_DIR/isaaclab.sh" ]; then
    echo -e "${YELLOW}Error: Isaac Lab not found at $ISAACLAB_DIR${NC}"
    exit 1
fi
```
- **Critical Issue:** Path is hardcoded to maxime's home directory (`/home/maxime/`)
- **Fix needed:** Should use `/home/ethan/IsaacLab` instead
- Checks if `isaaclab.sh` (Isaac Lab's launcher) exists before proceeding
- Exits with error code 1 if not found

#### Conda Activation:
```bash
source /data/home/maxime/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
```
- **Critical Issue:** Another hardcoded path to maxime's conda
- **Fix needed:** Should be `/home/ethan/miniconda3/etc/profile.d/conda.sh`
- Sources conda initialization script to make `conda` command available
- Activates `env_isaaclab` environment (should be `isaac_lab` after new installation)

#### Default Parameters:
```bash
TASK="Isaac-Velocity-Rough-Unitree-Go1-v0"    # Locomotion on rough terrain
NUM_ENVS=4096                                  # Max parallel training environments
MAX_ITERATIONS=1500                            # Training iterations (24 steps each)
HEADLESS="--headless"                          # Run without GUI (faster)
RESUME=""                                      # Don't resume by default
CHECKPOINT=""                                  # No specific checkpoint
```
- `TASK`: Isaac Lab gym task name (must be registered)
- `NUM_ENVS`: 4096 environments is the standard for this project
- `MAX_ITERATIONS`: 1500 × 24 steps ≈ 147M environment transitions
- `HEADLESS`: Training runs faster without graphics rendering
- `RESUME` & `CHECKPOINT`: Flags for resuming from prior training

#### Argument Parser Loop:
```bash
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)           # Override task
        --num_envs)       # Override environment count
        --max_iterations) # Override iterations
        --resume)         # Auto-detect and resume latest checkpoint
        --checkpoint)     # Resume from specific checkpoint path
        --gui)            # Enable GUI (removes --headless)
        --help)           # Show help
        *)
            echo "Unknown option: $1"
            exit 1
esac
```
- Parses all command-line arguments
- Examples:
  - `train.sh --num_envs 1024 --gui` — Train with 1024 envs and GUI enabled
  - `train.sh --resume` — Resume from latest checkpoint
  - `train.sh --checkpoint logs/rsl_rl/unitree_go1_rough/2025-11-16_10-30-00/model_500.pt` — Resume from specific checkpoint

#### Execution:
```bash
bash isaaclab.sh -p /home/maxime/my_go1_project/scripts/train_go1_walking.py \
    --task "$TASK" \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERATIONS" \
    $RESUME \
    $CHECKPOINT \
    $HEADLESS
```
- **Critical Issue:** Another hardcoded path (`/home/maxime/my_go1_project/...`)
- **Fix needed:** Should use absolute path to the actual project
- `isaaclab.sh`: Isaac Lab's wrapper that sets up Omniverse environment
- `-p`: Specifies Python script to run (`train_go1_walking.py`)
- Passes all configured parameters to the Python script

**Key Issue:** This script has 3 hardcoded paths to user "maxime". These must be updated to work in your environment.

---

### `test.sh` - Testing/Evaluation Wrapper Script

**Purpose:** Load a trained model checkpoint and evaluate it (with optional video recording).

#### Key Differences from train.sh:

**Task Selection:**
```bash
TASK="Isaac-Velocity-Rough-Unitree-Go1-Play-v0"  # "Play" variant = evaluation mode
```
- Uses `-Play-v0` suffix: this is a variant with:
  - 50 parallel environments (not 4096)
  - No noise or curriculum (pure evaluation)
  - Rewards still calculated but not used
  - Observation space same, but deterministic

**Auto-detect Latest Checkpoint:**
```bash
LOG_DIR="logs/rsl_rl/unitree_go1_rough"
if [ -z "$CHECKPOINT" ]; then
    if [ -d "$LOG_DIR" ]; then
        LATEST_RUN=$(ls -t "$LOG_DIR" | head -n 1)                    # Most recent timestamped dir
        LATEST_MODEL=$(ls -t "$LOG_DIR/$LATEST_RUN"/model_*.pt | head -n 1)  # Latest model file
        CHECKPOINT="$LATEST_MODEL"
```
- Automatically finds the most recent training run
- Finds the latest checkpoint in that run (highest model_*.pt number)
- Enables quick testing without specifying paths manually

**Checkpoint Validation:**
```bash
if [ -z "$CHECKPOINT" ]; then
    echo -e "${YELLOW}Error: No checkpoint found. Please specify with --checkpoint${NC}"
    exit 1
fi
if [ ! -f "$CHECKPOINT" ]; then
    echo -e "${YELLOW}Error: Checkpoint not found: $CHECKPOINT${NC}"
    exit 1
fi
```
- First check: was a checkpoint found/provided?
- Second check: does the file actually exist on disk?
- Both checks prevent running with invalid models

**Video Recording:**
```bash
--video                  # Flag to enable recording
--video_length 500      # Number of steps to record (500 steps = 10 seconds at 50 Hz)
```
- Records rendered view of one or more environments
- Useful for documentation and debugging

---

## 2. Python Scripts

### `train_go1_walking.py` - Main Training Script

**Purpose:** Orchestrate the entire training loop using Isaac Lab environment + RSL-RL PPO agent.

#### Imports & Setup:

```python
from isaaclab.app import AppLauncher
```
- Launches Omniverse (graphics and physics engine)
- Handles GPU selection, headless vs. GUI mode, etc.

```python
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
```
- `OnPolicyRunner`: RSL-RL training agent (PPO implementation)
- `ManagerBasedRLEnvCfg`: Isaac Lab environment config dataclass
- `RslRlVecEnvWrapper`: Adapter between Isaac Lab env and RSL-RL agent
- `RslRlOnPolicyRunnerCfg`: PPO hyperparameters (learning rate, network architecture, etc.)

```python
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
```
- Registers all available Isaac Lab tasks
- `parse_env_cfg`: Loads task-specific environment config from gym registry

#### CUDA Optimization:

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
```
- **allow_tf32**: Use TensorFlow 32-bit (mixed precision) for faster training
- **deterministic=False**: Allows non-deterministic CUDA ops (faster)
- **benchmark=False**: Disables cuDNN benchmarking (consistent speed)
- These trade perfect reproducibility for ~10-20% speed gain on A4500 GPUs

#### Main Function - Config Parsing:

```python
env_cfg = parse_env_cfg(
    args_cli.task, 
    device=args_cli.device, 
    num_envs=args_cli.num_envs, 
    use_fabric=not args_cli.cpu
)
```
- Loads the environment config (e.g., `Isaac-Velocity-Rough-Unitree-Go1-v0`)
- Overrides `num_envs`, `device` from command-line args
- `use_fabric`: Enables PyTorch Distributed (for multi-GPU training)

```python
agent_cfg_entry_point = gym.spec(args_cli.task).kwargs.get("rsl_rl_cfg_entry_point")
```
- Retrieves the entry point string from gym registry
- Example: `"go1_walking_env_cfg:RslRlPPOCfgGo1Rough"`
- This tells the script which config class to instantiate

```python
if isinstance(agent_cfg_entry_point, str):
    module_path, class_name = agent_cfg_entry_point.rsplit(":", 1)
    module = __import__(module_path, fromlist=[class_name])
    agent_cfg_class = getattr(module, class_name)
    agent_cfg = agent_cfg_class()
```
- Dynamically imports the config class
- Example: imports `RslRlPPOCfgGo1Rough` from `go1_walking_env_cfg` module
- Instantiates it to get actual hyperparameters

#### Command-Line Overrides:

```python
if args_cli.seed is not None:
    agent_cfg.seed = args_cli.seed
if args_cli.max_iterations is not None:
    agent_cfg.max_iterations = args_cli.max_iterations
```
- Allows command-line args to override config file values
- Example: `train_go1_walking.py --seed 42 --max_iterations 2000`
- Overrides take precedence over config file

#### Logging Directory Setup:

```python
log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
log_root_path = os.path.abspath(log_root_path)
```
- Creates path like: `logs/rsl_rl/unitree_go1_rough/`
- `experiment_name` comes from agent config
- `abspath`: Converts to absolute path to avoid issues with working directory

#### Resume/Checkpoint Logic:

```python
resume_path = None
if args_cli.checkpoint:
    resume_path = args_cli.checkpoint
elif args_cli.resume:
    # Find latest checkpoint
    run_dirs = sorted([d for d in os.listdir(log_root_path) 
                     if os.path.isdir(os.path.join(log_root_path, d))])
    latest_run = run_dirs[-1]
    checkpoints = [f for f in os.listdir(checkpoint_dir) 
                 if f.startswith("model_") and f.endswith(".pt")]
    latest_checkpoint = sorted(checkpoints, ...)[-1]
    resume_path = os.path.join(checkpoint_dir, latest_checkpoint)
```
- **Three cases:**
  1. Explicit checkpoint provided: use it
  2. `--resume` flag: auto-detect latest checkpoint in latest run
  3. Neither: start fresh training
- Latest checkpoint determined by sorting model_*.pt files numerically
- Old behavior: would start in new timestamp directory; new behavior should reuse same log_dir when resuming

#### Environment & Agent Creation:

```python
env = gym.make(args_cli.task, cfg=env_cfg)
env = RslRlVecEnvWrapper(env)
```
- `gym.make()`: Instantiates the Isaac Lab environment (4096 parallel sims)
- `RslRlVecEnvWrapper`: Converts Isaac Lab obs/action format to RSL-RL format
  - Input: Isaac Lab action dict → Output: flattened action vector
  - Input: Isaac Lab obs tensor → Output: flattened obs vector

```python
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
if resume_path:
    runner.load(resume_path)
```
- `OnPolicyRunner`: RSL-RL's training orchestrator
  - Manages actor/critic networks
  - Handles rollout buffer, advantage computation, gradient updates
- `.load()`: Restores network weights, optimizer state, training step count

#### Training Loop:

```python
runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
```
- **num_learning_iterations**: How many gradient update steps (each involves 4096 envs × 24 steps)
- **init_at_random_ep_len=True**: Randomly reset some environments each step (better gradient estimation)
- This is the core training loop; runs ~30-60 minutes for 1500 iterations on A4500

#### Cleanup:

```python
env.close()
simulation_app.close()
```
- Closes Isaac Lab environment and physics engine
- Critical for GPU memory release and Omniverse cleanup

---

### `play_go1_walking.py` - Evaluation/Testing Script

**Purpose:** Load a trained checkpoint and run the policy in evaluation mode (no training, optionally record video).

#### Key Differences from Training:

**Inference Mode:**
```python
runner.get_inference_policy()
```
- Extracts just the actor network in eval mode (no critic, no gradients)
- Much faster than training
- Typically runs at 1000+ steps/second on A4500

**No Learning:**
```python
with torch.no_grad():
    actions = inference_policy(obs)
```
- `torch.no_grad()`: Disables gradient computation (saves memory/speed)
- Only forward pass, no backprop

**Video Recording:**
```python
from PIL import Image
import imageio
video_writer = imageio.get_writer(
    os.path.join(video_dir, f"{video_name_prefix}.mp4"),
    fps=50 // args_cli.video_interval,
    codec='libx264'
)
```
- Captures frames at specified interval (e.g., every 5 steps → 10 FPS video)
- Encodes to MP4 using H.264 codec (standard for web/sharing)
- Example: `--video_length 500 --video_interval 5` → 100 frames → 10 sec video at 10 FPS

**Episode Statistics:**
```python
if dones.any():
    episode_count += dones.sum().item()
    avg_reward = episode_reward / episode_length
    print(f"[INFO] Episode {episode_count} | Steps: {episode_length} | Avg Reward: {avg_reward:.3f}")
```
- Tracks completed episodes
- Prints per-episode reward (used to assess convergence)
- In parallel eval with 50 envs, might complete 10+ episodes per second

#### Environment Reset:

```python
obs = env.reset()
if isinstance(obs, tuple):
    obs = obs[0]
```
- `env.reset()` may return `(obs, info)` tuple or just `obs`
- Extracts just the observation
- Handles both Isaac Lab and wrapped env formats

#### Simulation Loop:

```python
while simulation_app.is_running():
    actions = inference_policy(obs)
    obs, rewards, dones, infos = env.step(actions)
    
    if video_enabled and step_count % args_cli.video_interval == 0:
        frame = env.unwrapped.sim.render()
        video_writer.append_data(frame)
```
- Continuous loop while Omniverse app is running
- Step environment with policy action
- Capture frame if video recording enabled
- Stop after `video_length` steps if recording

---

## 3. Execution Flow

### Training Flow:
```
train.sh --num_envs 4096 --max_iterations 1500
    ↓
isaaclab.sh -p train_go1_walking.py
    ↓
train_go1_walking.py main()
    ├─ Load env config (go1_walking_env_cfg.py)
    ├─ Load agent config (RslRlPPOCfgGo1Rough)
    ├─ Create 4096 parallel Go1 robots in simulation
    ├─ Create RSL-RL PPO agent
    └─ Loop 1500 times:
        ├─ Collect 4096 envs × 24 steps = 98,304 transitions
        ├─ Compute advantages using learned critic
        ├─ Update actor/critic networks (multiple minibatch iterations)
        ├─ Save checkpoint every N steps
        └─ Log metrics (reward, loss, policy entropy)
    ↓
logs/rsl_rl/unitree_go1_rough/2025-11-16_XX-XX-XX/
    ├─ model_0.pt, model_100.pt, model_500.pt, ... (checkpoints)
    ├─ logs.csv (training metrics)
    └─ config.yaml (saved config)
```

### Evaluation Flow:
```
test.sh [--checkpoint PATH]
    ↓
isaaclab.sh -p play_go1_walking.py
    ↓
play_go1_walking.py main()
    ├─ Load env config (same as training)
    ├─ Find/load checkpoint (latest if not specified)
    ├─ Create 50 parallel Go1 robots (eval mode)
    ├─ Load trained actor network
    └─ Loop while simulation running:
        ├─ Step 50 envs with policy
        ├─ Record episode stats
        ├─ [Optional] Capture video frames
        └─ Print episode completion info
    ↓
videos/go1_policy_20251116_102030.mp4  (if --video enabled)
```

---

## 4. Command Reference

### Training Commands:

```bash
# Basic training (4096 envs, 1500 iterations, headless)
bash train.sh

# Training with GUI
bash train.sh --gui

# Custom environment count (use fewer for testing)
bash train.sh --num_envs 512

# Train for longer
bash train.sh --max_iterations 3000

# Resume from latest checkpoint
bash train.sh --resume

# Resume from specific checkpoint
bash train.sh --checkpoint logs/rsl_rl/unitree_go1_rough/2025-11-16_10-30-00/model_1000.pt

# Flat terrain variant (easier)
bash train.sh --task Isaac-Velocity-Flat-Unitree-Go1-v0
```

### Testing Commands:

```bash
# Test with latest checkpoint (auto-detected)
bash test.sh

# Test specific checkpoint
bash test.sh --checkpoint logs/rsl_rl/unitree_go1_rough/2025-11-16_10-30-00/model_1500.pt

# Test with video recording
bash test.sh --video

# Test with fewer environments (for debugging)
bash test.sh --num_envs 10

# Test with GUI
bash test.sh --gui

# Combined: test latest with video, 50 envs
bash test.sh --video --num_envs 50
```

---

## 5. Key Parameters

| Parameter | train.sh | test.sh | Meaning |
|-----------|----------|---------|---------|
| `--task` | Task name | Task name | Which gym task to run (rough/flat, training/play) |
| `--num_envs` | 4096 | 50 | Parallel environments (higher = faster training, higher VRAM) |
| `--max_iterations` | 1500 | — | Training steps (only for training) |
| `--resume` | Auto-detect | — | Resume from latest (only for training) |
| `--checkpoint` | Specify path | Specify/auto-detect | Model weights file |
| `--video` | — | Enabled/disabled | Record video of evaluation |
| `--video_length` | — | 500 | Steps to record |
| `--video_interval` | — | 250 | Frame skip (higher = faster playback) |
| `--gui` | Enabled/disabled | Enabled/disabled | Show GUI (slower, useful for debugging) |

---

## 6. Troubleshooting

### Issue: "Isaac Lab not found at /home/maxime/..."
**Cause:** Hardcoded paths in train.sh and test.sh
**Solution:** 
```bash
# Edit train.sh and test.sh, replace:
# /home/maxime/IsaacLab-main → /home/ethan/IsaacLab
# /data/home/maxime/miniconda3 → /home/ethan/miniconda3
# /home/maxime/my_go1_project → /home/ethan/go1-walking-policy
```

### Issue: "No checkpoints found"
**Cause:** Training hasn't run yet, or checkpoint in wrong directory
**Solution:**
```bash
# First run training
bash train.sh

# Wait for at least 1 iteration (takes ~10-20 seconds)
# Then run testing
bash test.sh
```

### Issue: "Out of memory" during training
**Cause:** 4096 environments exceed GPU VRAM
**Solution:**
```bash
# Reduce parallel environments
bash train.sh --num_envs 2048

# Or use CPU fallback (slower)
bash train.sh --num_envs 4096  # Script will handle auto-adjustment
```

### Issue: Training very slow
**Possible causes:**
1. GUI enabled (`--gui` flag) — disables headless mode
2. Too many environments for VRAM — reduce with `--num_envs`
3. CUDA version mismatch — verify with `nvidia-smi`

---

## 7. Integration with Go1 Robot

After training completes:
1. Export the checkpoint (`logs/rsl_rl/unitree_go1_rough/.../model_1500.pt`)
2. Run the Go1 robot communication module to stream observations
3. Load policy on robot's onboard computer
4. Stream real-time observations to policy, execute actions on robot

This is handled by `play_go1_web_streaming.py` (separate from this training pipeline).

