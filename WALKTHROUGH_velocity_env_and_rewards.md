# Detailed Line-by-Line Walkthrough: velocity_env_cfg.py and rewards.py

## Part 1: velocity_env_cfg.py — Base Locomotion Environment Configuration

**File location:** `config/base/velocity_env_cfg.py` (335 lines)
**Purpose:** Defines the complete base configuration for a legged robot locomotion environment that tracks velocity commands. This file is the "template" that `go1_walking_env_cfg.py` inherits from and customizes.

---

### Section 1: Imports and Setup (Lines 1–30)

```python
import math
from dataclasses import MISSING
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from mdp import *  # noqa: F401, F403
import mdp

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
```

**Explanation:**
- `import math` — Used for `math.sqrt()` in reward params.
- `MISSING` from dataclasses — A sentinel used in IsaacLab to mark required fields that must be overridden (e.g., `robot: ArticulationCfg = MISSING`).
- `sim_utils`, `ArticulationCfg`, etc. — IsaacLab core config and utility classes.
- Manager imports (`CurrTerm`, `EventTerm`, `ObsTerm`, `RewTerm`, `DoneTerm`) — These are aliases for config term types used to define rewards, terminations, observations, events, and curriculum.
- `InteractiveSceneCfg`, `ContactSensorCfg`, `RayCasterCfg` — Sensor and scene configuration classes.
- `sys.path.insert(...)` — Adds the config parent directory to Python path so we can import from `mdp/` (local MDP functions).
- `from mdp import *` and `import mdp` — Imports all MDP functions (rewards, terminations, etc.) from local `config/mdp/` directory.
- `ROUGH_TERRAINS_CFG` — Predefined terrain configuration (boxes, random rough, etc.) from IsaacLab.

---

### Section 2: Scene Configuration (Lines 35–82)

```python
@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/...",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    
    robot: ArticulationCfg = MISSING
    
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True
    )
    
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, ...),
    )
```

**Explanation:**
- **`MySceneCfg`** — Defines the static and dynamic elements of the simulation scene (world, terrain, sensors).
  
- **`terrain`** — Specifies the ground/terrain:
  - `prim_path="/World/ground"` → terrain is spawned at this path in the USD scene.
  - `terrain_type="generator"` → use procedural generation (not a pre-made mesh).
  - `terrain_generator=ROUGH_TERRAINS_CFG` → use IsaacLab's rough terrain generator (includes box obstacles, random rough sections, flat sections).
  - `max_init_terrain_level=5` → robots start at terrain difficulty level 5 (curriculum levels 0–5 exist; can progress beyond).
  - `collision_group=-1` → terrain collision layer (determines which objects collide with it).
  - `friction_combine_mode="multiply"` → friction is combined by multiplication (not addition).
  - `static_friction=1.0, dynamic_friction=1.0` → high friction to prevent sliding.
  - `visual_material` → appearance (marble tile texture).
  - `debug_vis=False` → don't show debug visualization of terrain.

- **`robot: ArticulationCfg = MISSING`** → Placeholder for robot asset. Must be set by subclass (e.g., Go1 overrides this with `UNITREE_GO1_CFG`).

- **`height_scanner`** — Ray-caster (LiDAR-like) sensor pointing downward from the robot to measure ground height:
  - `prim_path="{ENV_REGEX_NS}/Robot/base"` → sensor attached to the robot's base/trunk.
  - `offset=(0.0, 0.0, 20.0)` → ray origin is 20 meters above the robot body (ensures rays start above).
  - `ray_alignment="yaw"` → rays are cast in a grid aligned to robot's yaw (heading).
  - `pattern_cfg=GridPatternCfg(resolution=0.1, size=[1.6, 1.0])` → grid of rays with 0.1 m spacing, covering 1.6 m × 1.0 m area around the robot.
  - `mesh_prim_paths=["/World/ground"]` → rays only intersect the ground mesh.
  - Used in observations to give the policy terrain information ahead.

- **`contact_forces`** — Contact sensor that monitors collisions:
  - `prim_path="{ENV_REGEX_NS}/Robot/.*"` → attaches to all robot body parts (regex).
  - `history_length=3` → keeps last 3 frames of contact info (for temporal context).
  - `track_air_time=True` → also tracks how long each body has been in the air (used for feet-air-time reward).

- **`sky_light`** — Lighting for visualization (dome light with HDR texture).

---

### Section 3: Commands Configuration (Lines 88–105)

```python
@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi)
        ),
    )
```

**Explanation:**
- **`CommandsCfg`** — Defines how commands (targets) are generated for the policy.
  
- **`base_velocity`** — A velocity command generator:
  - `asset_name="robot"` → targets the "robot" asset in the scene.
  - `resampling_time_range=(10.0, 10.0)` → commands resampled every 10 seconds (constant in this case).
  - `rel_standing_envs=0.02` → 2% of environments are commanded to stand still (zero velocity).
  - `rel_heading_envs=1.0` → 100% of environments have yaw (heading) commands.
  - `heading_command=True` → enable yaw/heading commands (allow turning).
  - `heading_control_stiffness=0.5` → how strongly to enforce heading (0–1).
  - `debug_vis=True` → visualize command vectors in the simulator.
  - `ranges` → velocity command bounds (linear x, y: ±1.0 m/s; angular z: ±1.0 rad/s; heading: ±π radians).
  - During training, the environment randomly samples commands within these ranges and the policy learns to track them.

---

### Section 4: Actions Configuration (Lines 108–112)

```python
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
        use_default_offset=True
    )
```

**Explanation:**
- **`ActionsCfg`** — Defines what the policy outputs and how it maps to robot commands.
  
- **`joint_pos`** — Joint position command action type:
  - `asset_name="robot"` → targets robot joints.
  - `joint_names=[".*"]` → regex matching all joints (wildcard).
  - `scale=0.5` → policy output is scaled by 0.5 before applying to joints (prevents extreme commands).
  - `use_default_offset=True` → apply commands as offsets from default joint positions (not absolute angles).
  - The policy outputs a vector of size = number of joints, each value in [-1, 1], scaled by 0.5, added to default position.

---

### Section 5: Observations Configuration (Lines 115–175)

```python
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
```

**Explanation:**
- **`ObservationsCfg`** → Defines what the policy "sees" (state/observation vector).
  
- **`PolicyCfg`** — Inner class grouping observations used by the policy network:
  - Each `ObsTerm` specifies a piece of the observation vector:
    1. **`base_lin_vel`** — Robot's linear velocity (3D), with noise ±0.1 m/s (simulates sensor noise).
    2. **`base_ang_vel`** — Robot's angular velocity (3D), with noise ±0.2 rad/s.
    3. **`projected_gravity`** — Gravity vector in robot's frame (3D); indicates robot's orientation/roll/pitch.
    4. **`velocity_commands`** — Current velocity command target (3D + heading).
    5. **`joint_pos`** — Joint positions relative to default, with small noise.
    6. **`joint_vel`** — Joint velocities relative to nominal, with noise.
    7. **`actions`** — Last action executed (previous policy output).
    8. **`height_scan`** — Terrain height measurements from the height scanner (array of heights ahead of robot), clipped to ±1.0 m.
  
  - **Noise** `Unoise(n_min, n_max)` — Adds random noise during training to make policy robust to sensor imperfection.
  - **Order preserved** — observations are concatenated in order; total obs vector is these terms stacked.
  
  - **`enable_corruption=True`** — During training, add observation noise (disabled in PLAY config for clean evaluation).
  - **`concatenate_terms=True`** — Combine all obs terms into one vector (vs. keeping them separate).

---

### Section 6: Events Configuration (Lines 178–267)

```python
@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )
```

**Explanation:**
- **Events** are perturbations or resets triggered at different times:
  - **`mode="startup"`** — Runs once when environment initializes.
  - **`mode="reset"`** — Runs at each episode reset.
  - **`mode="interval"`** — Runs periodically during episodes.

- **Startup events:**
  - **`physics_material`** — Randomizes friction coefficients (slight variation, all envs get ~0.8 static / 0.6 dynamic).
  - **`add_base_mass`** — Adds ±5 kg to base mass to simulate payload variation.
  - **`base_com`** — Shifts center-of-mass by ±5 cm (x, y) / ±1 cm (z) to introduce imbalance.

- **Reset events:**
  - **`base_external_force_torque`** — At reset, can apply external force/torque (ranges set to 0 here → no application, but can be customized).
  - **`reset_base`** — Resets base pose (x, y, yaw ±0.5 m, ±π rad) and velocities (±0.5 m/s, ±0.5 rad/s).
  - **`reset_robot_joints`** — Resets joint positions/velocities to nominal (scale 0.5–1.5 → ±50% of default range).

- **Interval events:**
  - **`push_robot`** — Every 10–15 seconds, apply a random velocity push (±0.5 m/s in x/y) to simulate disturbance.

These events improve policy robustness by exposing it to variability at startup and during episodes.

---

### Section 7: Rewards Configuration (Lines 270–330)

```python
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.125,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    )
    
    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)
```

**Explanation:**
- Each `RewTerm` specifies:
  - **`func`** — The reward function (implemented in `config/mdp/rewards.py`).
  - **`weight`** — Coefficient (positive encourages behavior, negative penalizes).
  - **`params`** — Function-specific parameters.

- **Primary task rewards:**
  - **`track_lin_vel_xy_exp`** (weight=1.0) — Exponential reward for matching commanded linear velocity (x, y). `std=math.sqrt(0.25)` controls the reward sharpness (tighter → more penalized for errors).
  - **`track_ang_vel_z_exp`** (weight=0.5) — Exponential reward for matching angular velocity (yaw). Lower weight than linear.

- **Gait and stability rewards:**
  - **`feet_air_time`** (weight=0.125) — Reward long steps (time feet spend in air), filtered by contact sensor on foot bodies and command threshold.

- **Penalties:**
  - **`lin_vel_z_l2`** (weight=-2.0) — Penalize vertical motion (jumping).
  - **`ang_vel_xy_l2`** (weight=-0.05) — Penalize roll/pitch rotations (stay upright).
  - **`dof_torques_l2`** (weight=-1e-5) — Penalize high joint torques (energy efficiency).
  - **`dof_acc_l2`** (weight=-2.5e-7) — Penalize high joint accelerations (smooth movements).
  - **`action_rate_l2`** (weight=-0.01) — Penalize action changes (continuity).
  - **`undesired_contacts`** (weight=-1.0) — Penalize thigh collisions (if applicable).

- **Optional penalties (default weight=0.0):**
  - **`flat_orientation_l2`** — Can be enabled to penalize non-flat base (not priority in base config).
  - **`dof_pos_limits`** — Can be enabled to penalize joint limit violations.

---

### Section 8: Terminations Configuration (Lines 333–345)

```python
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
```

**Explanation:**
- **Episode terminations** — Conditions that end an episode early.
  
- **`time_out`** — Episode reaches maximum length (20 seconds in this project); `time_out=True` marks this as a time-based termination.
  
- **`base_contact`** — Episode ends if base body makes contact with terrain (detected by contact sensor). Indicates robot fell. `threshold=1.0` means any contact force > 1 N triggers termination.

---

### Section 9: Curriculum Configuration (Lines 348–353)

```python
@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
```

**Explanation:**
- **Curriculum learning** — Adaptive difficulty based on policy performance.
  
- **`terrain_levels`** — Uses the `terrain_levels_vel` function (in `config/mdp/curriculums.py`) to:
  - Increase terrain difficulty if robot walks far enough.
  - Decrease difficulty if robot doesn't travel the expected distance.
  - This gradually exposes the policy to harder terrain as it improves.

---

### Section 10: Main Environment Configuration (Lines 356–395)

```python
@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        
        # update sensor update periods
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
```

**Explanation:**
- **`LocomotionVelocityRoughEnvCfg`** — The main environment config combining all the above components.
  
- **Aggregated configs:**
  - `scene` — 4096 environments, 2.5 m spacing.
  - `observations`, `actions`, `commands`, `rewards`, `terminations`, `events`, `curriculum` — all the MDP layers.

- **Post-init adjustments:**
  - **`decimation = 4`** — Actions applied every 4 simulation steps (200 Hz / 4 = 50 Hz action frequency).
  - **`episode_length_s = 20.0`** — Episodes last 20 seconds → ~1000 actions per episode.
  - **`sim.dt = 0.005`** — Simulation timestep is 0.005 seconds (200 Hz).
  - **`render_interval = decimation`** — Render (visualize) every 4 steps.
  - **`gpu_max_rigid_patch_count`** — PhysX GPU memory allocation for collision patches.
  - **Sensor update periods:**
    - Height scanner updates every `decimation * dt = 4 * 0.005 = 0.02 s` (50 Hz, in sync with actions).
    - Contact sensor updates every `dt = 0.005 s` (200 Hz, matches simulation).
  - **Terrain curriculum:**
    - Checks if curriculum is defined; if so, enables curriculum in terrain generator.

---

## Part 2: rewards.py — Reward Function Implementations

**File location:** `config/mdp/rewards.py` (~150 lines)
**Purpose:** Implements individual reward functions called by the environment during training. These are referenced in `RewardsCfg` via `RewTerm(func=mdp.reward_name, ...)`.

---

### Overview of Reward Functions

The file contains approximately 6–8 main reward functions. Here's a line-by-line tour of a few key ones:

---

### Function: `feet_air_time()`

```python
def feet_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward
```

**Explanation:**
- **Purpose:** Reward the robot for lifting its feet (natural gait), but only when commanded to move.

- **Steps:**
  1. **`contact_sensor = env.scene.sensors[sensor_cfg.name]`** — Get the contact sensor object (monitors foot bodies).
  2. **`first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]`** — Boolean tensor indicating which feet just made contact (transition from air to ground).
  3. **`last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]`** — For each foot, how long it was in the air in the last cycle.
  4. **`reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)`** — For each foot:
     - If foot just made contact (`first_contact=True`, 1.0) AND `last_air_time > threshold`, reward = `last_air_time - threshold`.
     - Sum across all feet per environment → scalar reward per env.
  5. **`reward *= torch.norm(...) > 0.1`** — Zero out the reward if commanded velocity magnitude is < 0.1 m/s (don't reward air time when standing still).
  6. **`return reward`** — Shape: [num_envs], each value is the air-time reward for that environment.

---

### Function: `track_lin_vel_xy_yaw_frame_exp()`

```python
def track_lin_vel_xy_yaw_frame_exp(
    env,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)
```

**Explanation:**
- **Purpose:** Reward the robot for matching commanded linear velocity (x, y in the robot's frame).

- **Steps:**
  1. **`asset = env.scene[asset_cfg.name]`** — Get the robot asset.
  2. **`yaw_quat(asset.data.root_quat_w)`** — Extract the yaw (heading) as a quaternion.
  3. **`quat_apply_inverse(...)`** — Rotate the robot's world-frame linear velocity into the robot's yaw-aligned frame (so velocity is relative to robot heading).
  4. **`vel_yaw[:, :2]`** — Keep only x, y components.
  5. **`lin_vel_error = torch.sum(torch.square(command[:, :2] - vel_yaw[:, :2]), dim=1)`** — Squared error between commanded and actual velocity.
  6. **`torch.exp(-lin_vel_error / std**2)`** — Exponential reward:
     - If error=0 (perfect tracking) → reward=1.0.
     - As error increases, reward drops exponentially.
     - `std` parameter controls the slope (smaller std → steeper decline).

---

### Function: `track_ang_vel_z_world_exp()`

```python
def track_ang_vel_z_world_exp(
    env,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)
```

**Explanation:**
- **Purpose:** Reward for matching yaw (angular) velocity commands.

- Similar structure to linear velocity tracking:
  1. Get robot asset.
  2. Compute error: `(commanded_yaw_vel - actual_yaw_vel)^2` (note: no frame rotation needed; yaw is same in all frames).
  3. Return exponential reward based on error.

---

### Function: `lin_vel_z_l2()` (Penalty Example)

```python
def lin_vel_z_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize vertical motion (jumping)."""
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_w[:, 2])
```

**Explanation:**
- **Purpose:** Returns squared vertical velocity. Used with negative weight (e.g., -2.0) to penalize jumping.
- Simple L2 norm of vertical velocity component.

---

### Function: `action_rate_l2()`

```python
def action_rate_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize rapid action changes."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
```

**Explanation:**
- **Purpose:** Penalizes large differences between consecutive actions, encouraging smooth control.
- Computes squared norm of (current_action - previous_action), summed across all action dimensions.

---

### Function: `stand_still_joint_deviation_l1()`

```python
def stand_still_joint_deviation_l1(
    env,
    command_name: str,
    command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)
```

**Explanation:**
- **Purpose:** When robot is commanded to stand still (command norm < threshold), penalize joint deviations to encourage stable standing.
- Multiplies the joint deviation penalty by a boolean mask (1 if standing, 0 if moving).

---

### Key Design Patterns in rewards.py

1. **Function signature:**
   - Always takes `env: ManagerBasedRLEnv` as first arg.
   - Takes `command_name`, `sensor_cfg`, or `asset_cfg` as needed.
   - Returns `torch.Tensor` of shape `[num_envs]` (scalar reward per environment).

2. **Accessing robot state:**
   - `env.scene[asset_name]` → articulation asset (robot).
   - `asset.data.root_lin_vel_w` → linear velocity (world frame, shape [num_envs, 3]).
   - `asset.data.root_ang_vel_w` → angular velocity (world frame, shape [num_envs, 3]).
   - `asset.data.root_quat_w` → quaternion orientation (world frame).
   - `asset.data.joint_pos` → joint positions.
   - `asset.data.joint_vel` → joint velocities.

3. **Accessing command:**
   - `env.command_manager.get_command(command_name)` → shape [num_envs, command_dim].

4. **Accessing sensors:**
   - `env.scene.sensors[sensor_name]` → sensor object.
   - `contact_sensor.data.net_forces_w` → contact forces.
   - `contact_sensor.data.last_air_time` → time in air (for feet).

5. **Reward shaping:**
   - **Positive rewards** (weight > 0): encourage behavior.
   - **Negative rewards** (weight < 0): penalize behavior.
   - **Exponential shaping:** `torch.exp(-error/std^2)` for smooth, bounded rewards.
   - **L2 penalties:** `torch.square(value)` for energy or acceleration penalties.

6. **Masking/conditioning:**
   - Rewards often multiplied by boolean masks to apply conditionally (e.g., air-time reward only when moving).

---

## Summary: How These Files Interact

1. **`velocity_env_cfg.py`** defines the full environment structure:
   - Scene (terrain, sensors, robot placeholder).
   - Commands, actions, observations.
   - All reward and termination terms (by reference to functions).
   - Events and curriculum learning.

2. **`go1_walking_env_cfg.py`** inherits from velocity config and customizes:
   - Swaps in the Go1 robot asset.
   - Adjusts terrain scaling.
   - Tunes reward weights.
   - Customizes events for Go1 size/capabilities.

3. **`rewards.py`** implements the actual reward computations:
   - Each function is referenced in `RewardsCfg` by name.
   - Functions access robot state, sensors, and commands via the `env` object.
   - Return scalars per environment, which are multiplied by weights and summed into total reward.

Together, they form the complete environment specification: **what to observe** (observations), **what to do** (actions), **how to succeed** (rewards), **when to stop** (terminations), and **what challenges to face** (events, curriculum).

---

## Quick Reference: Key Numbers

- **Simulation:** dt = 0.005 s (200 Hz), decimation = 4 → action freq = 50 Hz.
- **Episodes:** 20 seconds → ~1000 actions/episode.
- **Training:** 4096 envs, 24 steps per env per iteration, 1500 iterations → ~147M transitions.
- **Rewards priority:**
  1. Velocity tracking (weights 1.0 and 0.5).
  2. Penalties on unwanted motion (vertical, roll/pitch, torques).
  3. Gait quality (feet air time, action continuity).
- **Terminations:**
  1. Time out (20 s).
  2. Trunk contact (fall).

