# ============================================================================
# Base Locomotion Environment Configuration for Velocity Tracking
# ============================================================================
#
# This file defines the base configuration for quadruped locomotion in Isaac Lab.
# It includes scene setup, MDP components (commands, actions, observations, rewards),
# and training parameters. This serves as the base class that specific robots
# (like Go1) inherit from and customize.
#
# Key Components:
#   1. Scene: Terrain, robot, sensors
#   2. Commands: Velocity commands for the policy to track
#   3. Actions: Joint position control
#   4. Observations: State information fed to the policy
#   5. Rewards: Incentive signals for learning
#   6. Terminations: Episode ending conditions
#   7. Events: Domain randomization and resets
#   8. Curriculum: Progressive training difficulty
#
# ============================================================================

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils  # type: ignore
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # type: ignore
from isaaclab.envs import ManagerBasedRLEnvCfg  # type: ignore
from isaaclab.managers import CurriculumTermCfg as CurrTerm  # type: ignore
from isaaclab.managers import EventTermCfg as EventTerm  # type: ignore
from isaaclab.managers import ObservationGroupCfg as ObsGroup  # type: ignore
from isaaclab.managers import ObservationTermCfg as ObsTerm  # type: ignore
from isaaclab.managers import RewardTermCfg as RewTerm  # type: ignore
from isaaclab.managers import SceneEntityCfg  # type: ignore
from isaaclab.managers import TerminationTermCfg as DoneTerm  # type: ignore
from isaaclab.scene import InteractiveSceneCfg  # type: ignore
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns  # type: ignore
from isaaclab.terrains import TerrainImporterCfg  # type: ignore
from isaaclab.utils import configclass  # type: ignore
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR  # type: ignore
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise  # type: ignore

# Import local MDP functions (reward, termination, event handlers)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from mdp import *  # noqa: F401, F403
import mdp

# Import pre-defined terrain configurations with increasing difficulty
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip  # type: ignore


# ============================================================================
# SCENE CONFIGURATION
# ============================================================================

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """
    Configuration for the 3D scene with terrain and robot.
    
    This defines:
    - Terrain: Procedurally generated rough terrain with varied features
    - Robot: Articulated figure (to be specified by subclass)
    - Sensors: Height scanner (terrain profile) and contact forces
    - Lighting: Dome light for realistic rendering
    """

    # ---- TERRAIN ----
    # Procedurally generated rough terrain with obstacles
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",  # Path in Isaac Sim scene hierarchy
        terrain_type="generator",    # Generate instead of import
        terrain_generator=ROUGH_TERRAINS_CFG,  # Use predefined terrain configs
        max_init_terrain_level=5,    # Start at difficulty level 5 (scale 0-10)
        collision_group=-1,          # Collision group for physics
        # Friction and restitution for terrain-foot interaction
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",      # How to combine frictions
            restitution_combine_mode="multiply",   # Bounce behavior
            static_friction=1.0,                   # Friction at rest
            dynamic_friction=1.0,                  # Friction while sliding
        ),
        # Visual appearance (marble brick texture)
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,  # Don't show debug visualization
    )

    # ---- ROBOT ----
    # Robot configuration (will be set by specific environment, e.g., Go1)
    robot: ArticulationCfg = MISSING

    # ---- SENSORS ----
    # Height scanner: Ray-caster that measures terrain height around the robot
    # Used to give the policy information about upcoming obstacles
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",  # Scan from robot base
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),  # Ray origin 20m above
        ray_alignment="yaw",  # Rays align with robot's yaw (heading)
        # Grid pattern: 0.1m resolution, 1.6m x 1.0m area
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],  # Scan this terrain mesh
    )

    # Contact sensor: Detects foot-ground contact for gait phase detection
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",  # Monitor all robot bodies
        history_length=3,       # Keep last 3 timesteps of contact data
        track_air_time=True,    # Track how long feet are in the air
    )

    # ---- LIGHTING ----
    # Dome light for realistic visualization
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,  # Brightness level
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


# ============================================================================
# MDP CONFIGURATION
# ============================================================================
# Markov Decision Process components: Commands, Actions, Observations, Rewards

@configclass
class CommandsCfg:
    """
    Configuration for velocity commands that the policy must track.
    
    The command generator produces:
    - Linear velocity (x, y)
    - Angular velocity (z/yaw)
    - Heading direction
    
    Commands are sampled at fixed intervals to provide target velocities.
    """

    # Uniform random velocity commands
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        # Resample commands every 10.0 seconds (constant during episode)
        resampling_time_range=(10.0, 10.0),
        # 2% of envs should be stationary (for training stability)
        rel_standing_envs=0.02,
        # 100% of envs use heading commands
        rel_heading_envs=1.0,
        # Enable explicit heading (angle) control
        heading_command=True,
        # Heading control stiffness: 0.5 = soft (less aggressive yaw tracking)
        heading_control_stiffness=0.5,
        debug_vis=True,  # Show command vectors for debugging
        # Command ranges (all normalized to [-1, 1])
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),   # Forward/backward: -1 to 1 m/s (speed depends on robot)
            lin_vel_y=(-1.0, 1.0),   # Left/right: -1 to 1 m/s
            ang_vel_z=(-1.0, 1.0),   # Spin: -1 to 1 rad/s
            heading=(-math.pi, math.pi)  # Facing direction in radians
        ),
    )


@configclass
class ActionsCfg:
    """
    Configuration for robot actions (motor commands).
    
    The policy outputs normalized joint position targets [-1, 1].
    These are scaled and applied to all robot joints.
    """

    # Joint position control: target joint angles
    # Scale: action * scale + default_offset = target_position
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],  # Control all joints (regex pattern)
        scale=0.5,  # Scale factor: 0.5 = smaller movements, safer training
        use_default_offset=True,  # Add default standing pose to actions
    )


@configclass
class ObservationsCfg:
    """
    Configuration for observations (state information) fed to the policy.
    
    Observations include:
    - Base motion (linear/angular velocity, orientation)
    - Commands (what the policy should track)
    - Joint state (positions and velocities)
    - Proprioception (last action, height scan)
    - Noise: Added for robustness to perception noise
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations used by the policy network."""

        # ---- INERTIAL MEASUREMENTS ----
        # Base (root) linear velocity in world frame [m/s]
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1)  # ±0.1 m/s noise
        )

        # Base angular velocity (roll, pitch, yaw rates) in body frame [rad/s]
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2)  # ±0.2 rad/s noise
        )

        # Gravity vector projected onto robot frame
        # Shows robot's tilt (3D orientation encoded as 3D vector)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05)  # ±0.05 tilt noise
        )

        # ---- COMMANDS ----
        # Current velocity commands to track (lin_x, lin_y, ang_z)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"}
        )

        # ---- JOINT STATE ----
        # Joint positions relative to default standing pose
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01)  # ±0.01 rad position noise
        )

        # Joint velocities (angular speeds) relative to default
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5)  # ±1.5 rad/s velocity noise
        )

        # ---- PROPRIOCEPTION ----
        # Last action taken (motor commands from previous step)
        # Helps policy understand recent control history
        actions = ObsTerm(func=mdp.last_action)

        # Height scan: Distance to terrain at 16 points around robot
        # Tells policy about upcoming terrain (obstacles, slopes)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),  # ±0.1 m terrain sensing noise
            clip=(-1.0, 1.0),  # Clamp to [-1, 1] range
        )

        def __post_init__(self):
            # Enable observation noise (for robustness training)
            self.enable_corruption = True
            # Concatenate all observation terms into single vector
            self.concatenate_terms = True

    # Create policy observation group
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """
    Configuration for domain randomization and reset events.
    
    Events apply physical randomization (startup) and reset episodic state (reset/interval).
    This improves robustness by training on diverse scenarios.
    """

    # ---- STARTUP EVENTS (Applied once at environment creation) ----

    # Randomize physics material: friction and restitution
    # Varies how slippery the terrain is and how bouncy the robot is
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),       # Friction when static
            "dynamic_friction_range": (0.6, 0.6),      # Friction when sliding
            "restitution_range": (0.0, 0.0),           # Bounciness (0=no bounce)
            "num_buckets": 64,  # Number of material variations
        },
    )

    # Randomize base (trunk) mass
    # Simulates robot with different payloads or battery states
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),  # ±5 kg variation
            "operation": "add",  # Add to base mass, don't replace
        },
    )

    # Randomize center of mass (CoM) offset
    # Simulates CoM shift from payload placement
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    # ---- RESET EVENTS (Applied when episode resets) ----

    # Apply random external force and torque to base
    # Simulates unexpected pushes or disturbances
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),    # No force applied (can be enabled)
            "torque_range": (-0.0, 0.0),  # No torque applied (can be enabled)
        },
    )

    # Reset robot base pose and velocity
    # Random initial position and orientation for each episode
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # Position ranges: uniform sampling within [-0.5, 0.5] meters
            "pose_range": {
                "x": (-0.5, 0.5),      # Left/right offset
                "y": (-0.5, 0.5),      # Forward/backward offset
                "yaw": (-3.14, 3.14),  # Rotation around vertical axis (full 360°)
            },
            # Velocity ranges: uniform random initial velocity
            "velocity_range": {
                "x": (-0.5, 0.5),      # Initial forward velocity [m/s]
                "y": (-0.5, 0.5),      # Initial side velocity [m/s]
                "z": (-0.5, 0.5),      # Initial vertical velocity [m/s]
                "roll": (-0.5, 0.5),   # Initial roll rate [rad/s]
                "pitch": (-0.5, 0.5),  # Initial pitch rate [rad/s]
                "yaw": (-0.5, 0.5),    # Initial yaw rate [rad/s]
            },
        },
    )

    # Reset joint positions and velocities
    # Randomize initial configuration for robustness
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            # Position scale: 0.5-1.5x default standing pose
            "position_range": (0.5, 1.5),
            # Velocity: start with zero joint velocity (legs not moving)
            "velocity_range": (0.0, 0.0),
        },
    )

    # ---- INTERVAL EVENTS (Applied periodically during episode) ----

    # Push robot at random intervals to simulate disturbances
    # Triggers every 10-15 seconds of simulated time
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),  # Push every 10-15 seconds
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),  # Random lateral push
                "y": (-0.5, 0.5),  # Random fore/aft push
            },
        },
    )


@configclass
class RewardsCfg:
    """
    Reward function configuration.
    
    The total reward is sum of weighted reward terms:
    R_total = sum(weight_i * reward_i)
    
    Positive weights encourage behavior, negative weights discourage it.
    """

    # ---- PRIMARY OBJECTIVES ----
    # Reward tracking linear velocity (x, y) using exponential kernel
    # Exponential: exp(-error^2 / std^2) gives smooth, continuous reward
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,  # Strong reward: main objective
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),  # std deviation for reward curve
        }
    )

    # Reward tracking angular velocity (yaw rotation) using exponential kernel
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,  # Secondary objective (0.5x less important than linear vel)
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
        }
    )

    # ---- PENALTIES FOR UNDESIRED BEHAVIORS ----

    # Penalize vertical motion (z velocity)
    # Strongly discourage bouncing up and down
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)

    # Penalize roll and pitch rotations
    # Encourage keeping robot upright (level ground)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    # Penalize high joint torques
    # Encourage energy-efficient movements, reduce stress on motors
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)

    # Penalize high joint accelerations
    # Smooth control reduces jerkiness and heat generation
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)

    # Penalize rapid action changes (smoothness penalty)
    # Encourage gradual control without sudden reversals
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # Reward feet spending time in the air (valid step detection)
    # Measures gait quality: feet should have clear swing phase
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.125,  # Moderate bonus for good stepping
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
            "command_name": "base_velocity",
            "threshold": 0.5,  # Minimum air time to get reward
        },
    )

    # Penalize undesired body contacts (e.g., hitting legs on ground)
    # Contact sensor monitors thigh bodies; touching terrain = penalty
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,  # Strong penalty for falling
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"),
            "threshold": 1.0,  # Contact force threshold for penalty
        },
    )

    # ---- OPTIONAL PENALTIES (Disabled by default: weight=0.0) ----

    # Penalize deviation from flat orientation (disabled: weight=0.0)
    # Can be enabled for robots that should stay perfectly level
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)

    # Penalize violating joint position limits (disabled: weight=0.0)
    # Joint limits enforced by physics; this is optional penalty
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)


@configclass
class TerminationsCfg:
    """
    Episode termination conditions.
    
    Episode ends when ANY termination condition is triggered.
    """

    # Episode timeout: max duration per episode
    # 20 seconds of simulated time (see __post_init__)
    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True  # This is an episode limit, not a failure
    )

    # Base contact termination (robot falls)
    # Episode ends if robot base (trunk) touches the ground
    # This is the primary failure condition
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
            "threshold": 1.0,  # Contact force threshold to trigger termination
        },
    )


@configclass
class CurriculumCfg:
    """
    Curriculum learning configuration.
    
    Progressively increases training difficulty over time.
    Helps policy learn from easy to hard scenarios.
    """

    # Terrain level progression: gradually increase difficulty
    # Maps episode count to terrain complexity level
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


# ============================================================================
# MAIN ENVIRONMENT CONFIGURATION
# ============================================================================

@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """
    Base configuration for legged locomotion with velocity tracking on rough terrain.
    
    This is the primary configuration class. Subclasses (e.g., for Go1) inherit
    and override specific components for robot-specific tuning.
    
    Key parameters:
    - 4096 parallel environments for efficient training
    - 20-second episodes
    - 5ms physics timestep (200 Hz simulation)
    - Curriculum learning to increase difficulty over time
    """

    # ---- SCENE AND ENTITIES ----
    # 4096 parallel environments: distributed across GPU for efficiency
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)

    # ---- MDP COMPONENTS ----
    observations: ObservationsCfg = ObservationsCfg()  # State observations
    actions: ActionsCfg = ActionsCfg()                 # Motor commands
    commands: CommandsCfg = CommandsCfg()              # Target velocities
    rewards: RewardsCfg = RewardsCfg()                 # Reward function
    terminations: TerminationsCfg = TerminationsCfg()  # Episode ending conditions
    events: EventCfg = EventCfg()                      # Domain randomization
    curriculum: CurriculumCfg = CurriculumCfg()        # Training difficulty progression

    def __post_init__(self):
        """Post-initialization configuration setup."""

        # ---- TIMING ----
        # Decimation: control update frequency relative to physics
        # decimation=4 means control every 4 physics steps
        # With sim.dt=0.005s, control frequency = 0.005*4 = 0.02s = 50 Hz
        self.decimation = 4

        # Episode length in seconds
        # 20 seconds = 20/0.02 = 1000 control steps per episode
        self.episode_length_s = 20.0

        # ---- SIMULATION PARAMETERS ----
        # Physics timestep: 0.005s = 5ms = 200 Hz simulation frequency
        # Small timesteps improve physics accuracy
        self.sim.dt = 0.005

        # Render interval: render every N physics steps
        # render_interval = decimation means render at control frequency
        self.sim.render_interval = self.decimation

        # Use terrain's physics material for ground
        self.sim.physics_material = self.scene.terrain.physics_material

        # PhysX GPU performance: max patch count for parallel collision detection
        # Tuned for systems with high parallelism
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # ---- SENSOR UPDATE PERIODS ----
        # Height scanner: update once per control step (4 physics steps)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # Contact sensor: update every physics step (no decimation) for accuracy
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # ---- CURRICULUM LEARNING SETUP ----
        # If terrain level curriculum is enabled, activate terrain generator curriculum
        # This makes terrain get progressively harder as training advances
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True  # Enable progressive difficulty
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False  # Disable: random difficulty
