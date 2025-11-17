# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ============================================================================
# Go1-Specific Environment Configuration
# ============================================================================
# This file defines custom environment configurations for the Unitree Go1 robot.
# It adapts the base locomotion velocity-tracking environment to Go1 parameters
# including terrain scale, action scaling, reward weights, and sensor configuration.
#

from isaaclab.utils import configclass  # type: ignore

# Import from local base configuration
from .base.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG  # isort: skip  # type: ignore


@configclass
class UnitreeGo1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """
    Go1-specific environment configuration for rough terrain training.
    
    This class inherits from LocomotionVelocityRoughEnvCfg and customizes:
    - Robot model (Unitree Go1)
    - Terrain scale (adjusted for Go1 size)
    - Action scaling (reduced for smooth movements)
    - Reward weights (tuned for Go1 walking)
    - Event parameters (mass randomization, resets)
    - Termination conditions (contact detection)
    """
    
    def __post_init__(self):
        # post init of parent: initializes all inherited fields to base defaults
        super().__post_init__()

        # ====== ROBOT AND SCENE ======
        # Set the robot model to Unitree Go1 (asset from isaaclab_assets)
        # {ENV_REGEX_NS} is a placeholder that expands to each environment's namespace
        # Example: /World/env_0/Robot, /World/env_1/Robot, etc.
        self.scene.robot = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # Point the height scanner (ray-caster) to the Go1 trunk for ground sensing
        # The height scanner is used in observations to detect terrain profile
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/trunk"
        
        # ====== TERRAIN SCALING ======
        # Go1 is smaller than default robots, so reduce terrain feature sizes
        # Boxes sub-terrain: heights between 2.5 cm and 10 cm (vs larger default)
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        
        # Random rough terrain: noise amplitude and step size for Go1 foot clearance
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # ====== ACTION SCALING ======
        # Reduce action scale to 0.25 for smoother, smaller joint movements
        # Prevents aggressive joint commands that could be unsafe or unrealistic
        self.actions.joint_pos.scale = 0.25

        # ====== EVENT CONFIGURATION ======
        # Disable random push events (external disturbances not needed for Go1)
        self.events.push_robot = None
        
        # Configure mass randomization to improve robustness across weight variations
        # Add between -1kg and +3kg to the trunk mass (simulates payload/battery variations)
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "trunk"
        
        # Configure external force/torque events to apply to the trunk body
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "trunk"
        
        # Reset joint positions with minimal randomization (1.0 = default scale)
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        
        # Define base pose reset ranges (position + orientation) and velocity ranges
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        
        # Disable center-of-mass randomization (handled separately via mass randomization)
        self.events.base_com = None

        # ====== REWARD CONFIGURATION ======
        # Primary objective: track commanded linear velocity (weight: 1.0 = strong)
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        
        # Secondary objective: track commanded angular velocity (turning, weight: 0.5)
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        
        # Gait quality rewards: encourage natural stepping pattern
        # Monitor feet air time using contact sensor on foot bodies (regex ".*_foot")
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.125
        
        # Disable undesired contact penalty (not applicable for Go1 thigh configuration)
        self.rewards.undesired_contacts = None
        
        # ====== PENALTIES FOR UNDESIRED BEHAVIORS ======
        # Strongly penalize vertical motion (z velocity) to keep robot flat
        self.rewards.lin_vel_z_l2.weight = -2.0
        
        # Penalize roll/pitch rotations to keep robot upright
        self.rewards.ang_vel_xy_l2.weight = -0.05
        
        # Penalize high joint torques to encourage efficient movements
        self.rewards.dof_torques_l2.weight = -1.0e-5
        
        # Penalize high joint accelerations to encourage smooth control
        self.rewards.dof_acc_l2.weight = -2.5e-7
        
        # Penalize rapid action changes to enforce continuity
        self.rewards.action_rate_l2.weight = -0.01
        
        # Optional: penalize deviations from flat (horizontal) orientation
        # This is set to a large negative value to strongly encourage keeping the base flat
        self.rewards.flat_orientation_l2.weight = -5.0
        
        # Disable joint position limit penalty (not enforcing limits as hard constraint)
        self.rewards.dof_pos_limits.weight = 0.0

        # ====== TERMINATION CONDITIONS ======
        # Episode terminates if the trunk body makes contact with terrain (fall detection)
        # This is the primary failure condition: if trunk touches ground, episode ends
        self.terminations.base_contact.params["sensor_cfg"].body_names = "trunk"


@configclass
class UnitreeGo1RoughEnvCfg_PLAY(UnitreeGo1RoughEnvCfg):
    """
    Go1 environment configuration optimized for policy evaluation/playback.
    
    Compared to UnitreeGo1RoughEnvCfg (training), this variant:
    - Reduces parallel environments to 50 (lower GPU memory)
    - Disables curriculum (no progressive terrain difficulty)
    - Disables observation corruption (shows true policy capability)
    - Removes external disturbances (deterministic evaluation)
    """
    
    def __post_init__(self):
        # post init of parent: inherit all training config settings first
        super().__post_init()

        # ====== ENVIRONMENT SIZING FOR EVALUATION ======
        # Use only 50 environments for testing (vs 4096 for training)
        # This reduces GPU memory usage and makes visualization/video recording feasible
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        
        # ====== TERRAIN RANDOMIZATION FOR PLAY ======
        # Allow random placement in terrain grid instead of curriculum-based levels
        # During play, we want diverse, unpredictable terrain for evaluation
        self.scene.terrain.max_init_terrain_level = None
        
        # Reduce terrain generator grid size to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            # Disable curriculum learning for play: terrain difficulty stays constant
            self.scene.terrain.terrain_generator.curriculum = False

        # ====== OBSERVATION AND SENSOR CONFIGURATION ======
        # Disable observation corruption (noise) to see actual policy behavior
        # During training, noise improves robustness; at test time, disable for clean evaluation
        self.observations.policy.enable_corruption = False
        
        # ====== EVENTS FOR PLAY ======
        # Remove external disturbances to keep evaluation stable and repeatable
        self.events.base_external_force_torque = None
        self.events.push_robot = None
