# Go1 Walking Policy — Complete Annotated Walkthrough Summary

This document summarizes all the detailed walkthroughs, annotations, and validation tools created to help you understand the Go1 walking policy project.

---

## What We've Covered

### 1. **Annotated go1_walking_env_cfg.py**

**Location:** `config/go1_walking_env_cfg.py`

**What changed:**
- Added comprehensive inline comments to every section explaining:
  - Robot and scene configuration
  - Terrain scaling for Go1 size
  - Action scaling to 0.25
  - Mass randomization and event parameters
  - Reward weight tuning (velocity tracking, penalties, optional terms)
  - Termination conditions (fall detection)
  - Play variant configuration (50 envs, no randomization, deterministic)

**Key sections with comments:**
- Imports and setup
- Class docstrings explaining purpose of each config class
- Robot and scene setup (UNITREE_GO1_CFG, height scanner, contact sensors)
- Terrain scaling block (boxes, random rough)
- Action scaling block
- Event configuration block (8+ events with detailed parameter explanations)
- Reward configuration block (velocity tracking, penalties, optional rewards)
- Termination configuration block
- Play variant with explanation of differences

**Use this for:** Quick reference to understand what each line of config does and why.

---

### 2. **Configuration Sanity Check Scripts**

#### a) `scripts/config_sanity_check.py`
**Purpose:** Full validation that runs inside IsaacLab environment.

**What it does:**
- Imports config classes (requires isaaclab packages).
- Instantiates `UnitreeGo1RoughEnvCfg` and `UnitreeGo1RoughEnvCfg_PLAY`.
- Validates key parameters (num_envs, scales, reward weights, etc.).
- Prints detailed output: scene config, reward functions, event config.

**Run inside Isaac Lab:**
```bash
cd /home/maxime/IsaacLab-main
conda activate env_isaaclab
cd /home/maxime/my_go1_project
python scripts/config_sanity_check.py
```

#### b) `scripts/config_validation_lightweight.py`
**Purpose:** Lightweight validation that requires NO dependencies.

**What it does:**
- Checks all required config files exist.
- Validates Python syntax in config files (no imports needed).
- Verifies expected symbols/classes are present.
- Checks reward term configuration structure.

**Run anytime:**
```bash
cd /home/ethan/go1-walking-policy
python3 scripts/config_validation_lightweight.py
```

**Output:** Shows file structure, syntax validity, and configuration completeness (we got a ✓ PASSED).

---

### 3. **velocity_env_cfg.py — Line-by-Line Walkthrough**

**Location:** `WALKTHROUGH_velocity_env_and_rewards.md` (created in this session)

**Sections covered:**

1. **Imports (Lines 1–30)**
   - What each import does (sim utils, managers, sensors, MDP functions).
   - How local MDP functions are imported via sys.path.

2. **Scene Configuration (Lines 35–82)**
   - Terrain setup: generator-based rough terrain, friction, visual materials.
   - Height scanner: ray-casting sensor for terrain perception.
   - Contact sensor: detects foot/body collisions and air time.
   - Lighting and visualization setup.

3. **Commands Configuration (Lines 88–105)**
   - Velocity command generator: resampling, standing/moving mix, velocity ranges.
   - How the policy is given random targets to track.

4. **Actions Configuration (Lines 108–112)**
   - Joint position commands: action scaling, default offset mode.
   - Maps policy outputs to joint angles.

5. **Observations Configuration (Lines 115–175)**
   - 8 observation terms: linear/angular velocity, gravity, commands, joint state, height scan.
   - Observation noise for training robustness.
   - Concatenation and corruption settings.

6. **Events Configuration (Lines 178–267)**
   - Startup events: randomize physics, mass, COM.
   - Reset events: pose/velocity resets, joint resets.
   - Interval events: periodic disturbances (push_robot).
   - Purpose: improve policy generalization and robustness.

7. **Rewards Configuration (Lines 270–330)**
   - Primary: velocity tracking (linear and angular).
   - Gait: feet air time.
   - Penalties: vertical motion, torques, accelerations, action rate, roll/pitch.
   - Optional: flat orientation, joint limits.
   - Weights and parameters for each term.

8. **Terminations Configuration (Lines 333–345)**
   - Time out (20 seconds).
   - Illegal contact (fall detection).

9. **Curriculum Configuration (Lines 348–353)**
   - Terrain level adjustment based on policy performance.

10. **Main Environment Config (Lines 356–395)**
    - Combines all above into `LocomotionVelocityRoughEnvCfg`.
    - Post-init: simulation dt, decimation, sensor update periods.
    - Terrain curriculum enabling/disabling logic.

---

### 4. **rewards.py — Line-by-Line Walkthrough**

**Covered in:** `WALKTHROUGH_velocity_env_and_rewards.md`

**Key reward functions explained:**

1. **feet_air_time()**
   - Rewards long steps (feet in air) when robot moves.
   - Uses contact sensor to track air time.
   - Masked by command threshold (no reward if standing still).

2. **track_lin_vel_xy_yaw_frame_exp()**
   - Exponential reward for linear velocity tracking (x, y).
   - Rotates world velocity into robot's yaw frame.
   - Error → exponential penalty via `exp(-error/std^2)`.

3. **track_ang_vel_z_world_exp()**
   - Exponential reward for yaw (turning) velocity tracking.
   - Similar exponential shaping.

4. **lin_vel_z_l2()**
   - Squared vertical velocity (penalty when weight < 0).
   - Discourages jumping.

5. **action_rate_l2()**
   - Squared norm of action differences (current - previous).
   - Encourages smooth control.

6. **stand_still_joint_deviation_l1()**
   - Conditional penalty: applies when command is near zero.
   - Encourages stable standing.

**Design patterns:**
- Function signature: `func(env, params...) → torch.Tensor[num_envs]`
- Access robot state via `env.scene[asset_name].data.*`
- Access commands via `env.command_manager.get_command(...)`
- Access sensors via `env.scene.sensors[name]`
- Use exponential/L2 shaping for smooth reward gradients.
- Mask rewards conditionally (e.g., only when moving).

---

## File Locations of Key Documents

| File | Purpose |
|------|---------|
| `config/go1_walking_env_cfg.py` | Go1-specific config (NOW WITH INLINE COMMENTS) |
| `config/base/velocity_env_cfg.py` | Base locomotion config (reference in walkthrough doc) |
| `config/mdp/rewards.py` | Reward function implementations (explained in walkthrough) |
| `scripts/config_sanity_check.py` | Full validation (needs IsaacLab) |
| `scripts/config_validation_lightweight.py` | Lightweight syntax/structure check (no deps) |
| `WALKTHROUGH_velocity_env_and_rewards.md` | Detailed line-by-line walkthroughs (THIS SESSION) |

---

## Quick Testing Commands

**Test without IsaacLab (fast, no dependencies):**
```bash
cd /home/ethan/go1-walking-policy
python3 scripts/config_validation_lightweight.py
```
Expected: ✓ All checks pass, confirming file structure and syntax.

**Test with IsaacLab (full validation inside Isaac Lab environment):**
```bash
cd /home/maxime/IsaacLab-main
conda activate env_isaaclab
cd /home/maxime/my_go1_project
python scripts/config_sanity_check.py
```
Expected: ✓ All config values match expected (4096 envs, 0.25 scale, reward weights, etc.).

---

## Key Numbers to Remember

| Parameter | Value | Meaning |
|-----------|-------|---------|
| Simulation dt | 0.005 s | 200 Hz physics |
| Decimation | 4 | Action frequency = 50 Hz |
| Episode length | 20 s | ~1000 actions/episode |
| Training envs | 4096 | Parallel environments |
| Training iterations | 1500 | Total training steps |
| Training data | ~147M | (1500 × 24 × 4096) transitions |
| Play envs | 50 | For evaluation (lower memory) |
| Action scale | 0.25 | Joint position offset range |
| Velocity reward weight | 1.0 | Primary objective |
| Vertical motion penalty | -2.0 | Strong disincentive |
| Torque penalty | -1e-5 | Weak energy penalty |

---

## How Everything Fits Together

1. **velocity_env_cfg.py** defines the template:
   - Scene: terrain, sensors, robot placeholder.
   - Observations: state vector (velocity, orientation, height, etc.).
   - Actions: joint position commands.
   - Commands: random velocity targets.
   - Rewards: 10+ terms weighted and combined.
   - Terminations: end episode on timeout or fall.
   - Events: randomization for robustness.
   - Curriculum: progressive terrain difficulty.

2. **go1_walking_env_cfg.py** customizes for Go1:
   - Swap in UNITREE_GO1_CFG robot asset.
   - Scale terrain/actions to Go1 size.
   - Tune reward weights (1.0 lin_vel, 0.5 ang_vel, -2.0 vertical, etc.).
   - Adjust event parameters (mass range -1 to +3 kg, resets, etc.).
   - Define PLAY variant for evaluation (50 envs, no noise, no disturbances).

3. **rewards.py** implements the math:
   - Each reward function accesses robot state, sensors, commands via the `env` object.
   - Returns scalar per environment.
   - IsaacLab multiplies by weight and sums into total reward.

4. **train_go1_walking.py** brings it together:
   - Loads config via gym registry (entry point defined in `config/__init__.py`).
   - Instantiates environment with config.
   - Creates RSL-RL OnPolicyRunner with PPO agent config.
   - Runs training loop: collect data, compute rewards, update policy.
   - Saves checkpoints.

---

## Next Steps for You

### Understanding
- ✅ Read the annotated `go1_walking_env_cfg.py` to see what each parameter does.
- ✅ Review `WALKTHROUGH_velocity_env_and_rewards.md` for deep dives into base config and rewards.
- ✅ Run `config_validation_lightweight.py` anytime to verify structure (no dependencies).

### Experimenting
1. **Small tuning:** Adjust reward weights in `go1_walking_env_cfg.py`, re-run training.
2. **Add reward:** Edit `config/mdp/rewards.py` to add new reward function, reference it in `go1_walking_env_cfg.py`.
3. **Change terrain:** Adjust `sub_terrains` noise/height ranges in `go1_walking_env_cfg.py`.
4. **Change hyperparams:** Edit `config/agents/rsl_rl_ppo_cfg.py` (learning rate, network size, iterations).

### Running
```bash
# Training
cd /home/maxime/IsaacLab-main
bash isaaclab.sh -p /home/maxime/my_go1_project/scripts/train_go1_walking.py \
    --task Isaac-Velocity-Rough-Unitree-Go1-Custom-v0 \
    --num_envs 4096 \
    --max_iterations 1500 \
    --headless

# Testing
bash isaaclab.sh -p /home/maxime/my_go1_project/scripts/play_go1_walking.py \
    --task Isaac-Velocity-Rough-Unitree-Go1-Custom-Play-v0 \
    --checkpoint logs/rsl_rl/unitree_go1_rough/<timestamp>/model_<iteration>.pt

# Monitoring
tensorboard --logdir logs/rsl_rl/
```

---

## Files Modified/Created This Session

1. **config/go1_walking_env_cfg.py** — Added comprehensive inline comments.
2. **scripts/config_sanity_check.py** — Created full validation script.
3. **scripts/config_validation_lightweight.py** — Created lightweight syntax check.
4. **WALKTHROUGH_velocity_env_and_rewards.md** — Created detailed walkthrough of base config and rewards.

All files are ready to use and well-commented for future reference.

