# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
═══════════════════════════════════════════════════════════════════════════════
                    REWARD FUNCTIONS FOR GO1 LOCOMOTION
═══════════════════════════════════════════════════════════════════════════════

Common functions that can be used to define rewards for the learning environment.

These reward functions form the OBJECTIVE FUNCTION that the PPO agent learns to
maximize during training. Each function computes a scalar reward (or batch of rewards
for parallel environments) that guides the learning process.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.

WHY REWARD SHAPING MATTERS:
──────────────────────────
In reinforcement learning, the reward function is the PRIMARY WAY we communicate
what behavior we want. The agent will learn to do whatever maximizes the cumulative
reward, regardless of our intentions. Therefore, careful reward design is critical.

Good reward functions:
1. Are dense (give feedback frequently, not just at episode end)
2. Are balanced (no single term dominates)
3. Use smooth functions (e.g., exponential) to provide gradients
4. Penalize undesired side effects (energy, instability)
5. Reward desired behaviors (tracking, smooth motion)

DESIGN PHILOSOPHY:
─────────────────
Primary Rewards (weight ≈ 1.0):
  • Track linear velocity commands (what human asks for)
  • Track angular velocity commands (rotation)

Secondary Rewards (weight ≈ 0.5):
  • Smooth foot motion (gait quality)
  • Energy efficiency penalties

Constraints/Penalties (weight < 0):
  • Penalize vertical motion (should move horizontally)
  • Penalize falling
  • Penalize high torques (energy)
  • Penalize jerky motions (smoothness)

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ═══════════════════════════════════════════════════════════════════════════════
# REWARD FUNCTION 1: FEET AIR TIME
# ═══════════════════════════════════════════════════════════════════════════════

def feet_air_time(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    sensor_cfg: SceneEntityCfg, 
    threshold: float
) -> torch.Tensor:
    """
    ═════════════════════════════════════════════════════════════════════════════
    Reward long steps taken by the feet using L2-kernel.
    ═════════════════════════════════════════════════════════════════════════════

    PURPOSE:
    ───────
    This function rewards the agent for taking steps that are longer than a
    threshold. This helps ensure that the robot lifts its feet off the ground
    and takes SMOOTH, COORDINATED steps rather than shuffling.

    The reward is computed as the sum of the time for which the feet are in
    the air. Longer air time = better gait quality.

    PHYSICS INSIGHT:
    ───────────────
    For quadrupeds:
    - Trotting gait: diagonal pairs move together → good for speed
    - Pacing gait: lateral pairs move together → good for stability
    - Crawling: one leg at a time → slow but very stable
    
    This reward encourages STRONG PHASE SEPARATION between legs.

    PARAMETERS:
    ──────────
    env : ManagerBasedRLEnv
        The environment object containing sensors, commands, and state
    
    command_name : str
        Name of the command buffer (e.g., "base_velocity_target")
        We use this to check if the robot SHOULD be moving
    
    sensor_cfg : SceneEntityCfg
        Configuration specifying which contact sensor to use and which
        body indices correspond to feet (e.g., feet indices [0,1,2,3])
    
    threshold : float
        Minimum air time in seconds to get reward
        Example: threshold=0.3 means feet must be off ground for 0.3+ seconds
        to receive reward. Values 0.2-0.5 seconds are typical for quadrupeds.

    RETURN:
    ──────
    torch.Tensor of shape (num_envs,)
        Reward for each environment. Positive values for good gaits,
        zero if robot not moving, negative if no lifting.

    MATHEMATICAL FORMULATION:
    ────────────────────────
    For each foot f in feet:
        reward_f = (last_air_time_f - threshold) if foot_made_first_contact
                 = 0 otherwise
    
    total_reward = sum(reward_f for all feet)
    
    Then apply command mask:
        if ||command_xy|| > 0.1: return total_reward
        else: return 0  (don't reward air time if standing still)

    EXAMPLE WALKTHROUGH:
    ───────────────────
    At timestep t:
    - Front-left foot: last_air_time = 0.4s, made contact → reward += (0.4-0.3) = 0.1
    - Front-right foot: last_air_time = 0.35s, made contact → reward += (0.35-0.3) = 0.05
    - Back-left foot: still in air → reward += 0
    - Back-right foot: still in air → reward += 0
    
    Total for this step: 0.15 (for that environment)
    
    Over whole episode (1000 steps), good gaits accumulate high reward.

    HYPERPARAMETER TUNING:
    ────────────────────
    If robot shuffles (doesn't lift feet):
        → Increase weight of this reward
        → Decrease threshold (easier to satisfy)
    
    If robot bounces too much (takes excessively high steps):
        → Decrease weight
        → Increase threshold (harder to satisfy)
    
    Typical values:
        threshold=0.2-0.5 seconds for trotting
        weight=0.5-2.0 in reward configuration
    """
    
    # Extract the contact sensor object from the scene
    # This sensor tracks which body parts are touching the ground
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # Compute which feet made their FIRST CONTACT with ground
    # These are the feet that just transitioned from air to ground
    # This is important: we reward COMPLETING a step, not just lifting
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    # Shape: (num_envs, num_feet) - binary: 1 if foot just touched ground, 0 otherwise
    
    # Get the time since last contact for each foot
    # This accumulates: 0.0 when touching, increases when in air
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    # Shape: (num_envs, num_feet) - seconds in air
    
    # Compute reward: (air_time - threshold) * first_contact
    # Only reward if foot just made contact AND air_time exceeded threshold
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # Shape: (num_envs,) - sum of air time achievements across all feet
    
    # CRITICAL: Don't reward air time if the robot isn't supposed to move
    # This prevents the agent from wasting energy bouncing when standing still
    command = env.command_manager.get_command(command_name)
    # command shape: (num_envs, 3) - [lin_vel_x, lin_vel_y, ang_vel_z]
    
    # Check command magnitude: ||[vx, vy]|| > 0.1 m/s threshold
    # If command is very small, robot should stand still, so no air time reward
    moving = torch.norm(command[:, :2], dim=1) > 0.1
    reward = reward * moving
    # Now reward is only nonzero when robot is actively moving
    
    return reward


# ═══════════════════════════════════════════════════════════════════════════════
# REWARD FUNCTION 2: FEET AIR TIME (BIPED VARIANT)
# ═══════════════════════════════════════════════════════════════════════════════

def feet_air_time_positive_biped(
    env, 
    command_name: str, 
    threshold: float, 
    sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    ═════════════════════════════════════════════════════════════════════════════
    Reward long steps taken by the feet for bipeds (two-legged robots).
    ═════════════════════════════════════════════════════════════════════════════

    PURPOSE:
    ───────
    Similar to feet_air_time but SPECIFICALLY DESIGNED FOR BIPEDS.
    
    For bipeds (humans, humanoids), we must ensure:
    1. Only ONE leg is in the air at a time (can't hop on both legs)
    2. The other leg provides stance support
    3. Single-leg stance time is maximized (reduces impact)

    This function ensures a proper WALKING GAIT, not hopping or crawling.

    BIPED PHYSICS:
    ──────────────
    Proper biped walking requires:
    - Single-leg stance: one leg on ground supporting body weight
    - Swing phase: other leg in air moving forward
    - Double-support: brief period when both touch (reduces impact)
    
    This reward encourages long single-support phases, which is biomechanically
    efficient and reduces energy cost.

    PARAMETERS:
    Same as feet_air_time, but designed for bipeds
    - threshold: max single-leg stance time to encourage (0.3-0.6s typical)

    RETURN:
    torch.Tensor of shape (num_envs,)

    ALGORITHM EXPLANATION:
    ────────────────────
    1. For each foot, compute time in current state:
       - If in contact: use contact_time
       - If in air: use air_time
    
    2. Identify "single stance" condition:
       - Count feet in contact
       - Single stance = exactly 1 foot touching ground
    
    3. Reward = minimum of in_mode_time across both feet,
       but ONLY when single_stance is true
    
    4. Clamp maximum to threshold (don't reward excessively long)
    
    5. Apply command mask (only when moving)

    EXAMPLE:
    ──────
    Timestep analysis:
    - Left foot: contact, in contact for 0.4s
    - Right foot: air, in air for 0.3s
    - Condition: exactly 1 in contact ✓ single_stance = true
    - Reward = min(0.4, 0.3) = 0.3s (clamped to threshold)
    
    Next timestep (both feet brief double support):
    - Left foot: contact, in contact for 0.01s
    - Right foot: contact, in contact for 0.01s
    - Condition: 2 in contact ✗ single_stance = false
    - Reward = 0 (penalize double support)
    """
    
    # Get contact sensor
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # Current air time for each foot (time since last contact)
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    # Shape: (num_envs, 2) for bipeds
    
    # Current contact time for each foot (time of current contact)
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    # Shape: (num_envs, 2)
    
    # Check which feet are currently in contact
    in_contact = contact_time > 0.0  # Binary: 1 if touching, 0 if not
    # Shape: (num_envs, 2) - boolean array
    
    # For each foot, get the time in its current mode:
    # - If in contact: use contact_time
    # - If in air: use air_time
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    # Shape: (num_envs, 2) - time in current phase for each foot
    
    # Check single-leg stance condition: exactly 1 foot in contact
    num_contacts = torch.sum(in_contact.int(), dim=1)  # 0, 1, or 2
    single_stance = num_contacts == 1
    # Shape: (num_envs,) - boolean
    
    # Compute reward:
    # - For single stance: take minimum time (controls cycle time)
    # - For double support/swing: reward is 0
    reward = torch.min(
        torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0),
        dim=1
    )[0]
    # Shape: (num_envs,)
    # Note: unsqueeze(-1) broadcasts single_stance for per-foot comparison
    
    # Clamp to maximum threshold (don't reward excessively long stances)
    # This prevents the agent from just standing on one leg doing nothing
    reward = torch.clamp(reward, max=threshold)
    # Now reward is in range [0, threshold]
    
    # Apply command mask: only reward during active movement
    command = env.command_manager.get_command(command_name)
    moving = torch.norm(command[:, :2], dim=1) > 0.1
    reward = reward * moving
    
    return reward


# ═══════════════════════════════════════════════════════════════════════════════
# REWARD FUNCTION 3: FEET SLIDE PENALTY
# ═══════════════════════════════════════════════════════════════════════════════

def feet_slide(
    env, 
    sensor_cfg: SceneEntityCfg, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    ═════════════════════════════════════════════════════════════════════════════
    Penalize feet sliding on the ground.
    ═════════════════════════════════════════════════════════════════════════════

    PURPOSE:
    ───────
    This function PENALIZES (negative reward) the agent for sliding its feet
    along the ground when they're in contact. Sliding indicates:
    
    1. Poor joint control (not holding position)
    2. Inefficient gait (losing traction)
    3. Potential for slipping in real world
    4. Wasted energy on friction

    Good locomotion requires DISCRETE foot placements, not continuous sliding.
    Think of proper walking: foot lands → stays firm → lifts → repeat
    Sliding: foot lands → drags across ground → bad

    PHYSICS INSIGHT:
    ───────────────
    Contact friction is a key challenge in legged locomotion:
    - Dry friction μ ≈ 0.5-1.5 (good grip)
    - Sliding reduces effective friction
    - Real robots have limited friction → must minimize sliding
    
    This penalty encourages discrete steps with minimal lateral motion.

    PARAMETERS:
    ──────────
    env : ManagerBasedRLEnv
        Environment
    
    sensor_cfg : SceneEntityCfg
        Contact sensor configuration (specifies which feet to monitor)
    
    asset_cfg : SceneEntityCfg
        Robot asset configuration (default "robot")
        Used to get body velocity data

    RETURN:
    torch.Tensor of shape (num_envs,)
        NEGATIVE reward (penalty) for sliding motion
        More sliding = larger negative value

    MATHEMATICAL FORMULATION:
    ────────────────────────
    For each foot f:
        if foot is in contact with ground:
            penalty_f = ||velocity_xy of foot f||
        else:
            penalty_f = 0
    
    total_penalty = sum(penalty_f for all feet)
    
    Note: Result is summed without negation here, but in config it's
    typically given weight < 0 to convert to penalty.

    IMPLEMENTATION DETAILS:
    ─────────────────────
    1. Get contact forces over recent history (5 timesteps)
    2. Max over time → identifies if foot made contact
    3. If contact force > 1.0 N (threshold), foot is in contact
    4. Get foot velocity in xy plane (horizontal)
    5. Multiply velocity by contact binary mask
    6. Sum across all feet and time

    EXAMPLE:
    ──────
    At timestep t:
    - Front-left foot: in contact, velocity = 0.05 m/s → penalty += 0.05
    - Front-right foot: in air, velocity = 0.3 m/s → penalty += 0 (not in contact)
    - Back-left foot: in contact, velocity = 0.02 m/s → penalty += 0.02
    - Back-right foot: in contact, velocity = 0.03 m/s → penalty += 0.03
    
    Total penalty for this step = 0.10 m/s
    Over episode: penalizes any sliding, encourages clean foot placements

    HYPERPARAMETER NOTES:
    ───────────────────
    Weight in config (typically -0.5 to -2.0):
        - Higher magnitude = stronger penalty on sliding
        - Increase if robot slides too much
        - Decrease if footsteps become too abrupt
    
    Threshold (1.0 N for contact detection):
        - Set to robot's foot contact force range
        - Go1: ~5-50 N per foot → 1.0 N is very light contact
    """
    
    # Get contact sensor
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # Get contact force history (past 5 timesteps)
    # This gives us a history of forces to determine if foot was in contact
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    # Shape: (num_envs, history_length=5, num_feet, 3 [fx, fy, fz])
    
    # Take norm of forces (magnitude), then max over time
    # Max over time identifies if foot made contact in recent history
    contact_magnitude = contact_forces.norm(dim=-1)  # Remove xyz dim
    # Shape: (num_envs, 5, num_feet)
    
    max_contact = contact_magnitude.max(dim=1)[0]
    # Shape: (num_envs, num_feet)
    
    # Binary mask: is foot in contact (force > 1.0 N)?
    contacts = max_contact > 1.0
    # Shape: (num_envs, num_feet)
    
    # Get robot asset
    asset = env.scene[asset_cfg.name]
    
    # Get body linear velocities in world frame, xy component only
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    # Shape: (num_envs, num_feet, 2) - vx, vy for each foot
    
    # Compute magnitude of xy velocity (how fast foot is sliding horizontally)
    vel_magnitude = body_vel.norm(dim=-1)
    # Shape: (num_envs, num_feet)
    
    # Apply contact mask: only penalize velocity if foot is in contact
    penalized_vel = vel_magnitude * contacts.float()
    # Shape: (num_envs, num_feet)
    
    # Sum across all feet
    reward = torch.sum(penalized_vel, dim=1)
    # Shape: (num_envs,)
    
    # Note: This returns positive values that will be weighted negative
    # in the config (e.g., weight=-0.5) to become a penalty
    
    return reward


# ═══════════════════════════════════════════════════════════════════════════════
# REWARD FUNCTION 4: TRACK LINEAR VELOCITY (PRIMARY OBJECTIVE)
# ═══════════════════════════════════════════════════════════════════════════════

def track_lin_vel_xy_yaw_frame_exp(
    env, 
    std: float, 
    command_name: str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    ═════════════════════════════════════════════════════════════════════════════
    Reward tracking of linear velocity commands (xy axes) in the gravity-aligned
    robot frame using exponential kernel.
    ═════════════════════════════════════════════════════════════════════════════

    PURPOSE:
    ───────
    This is the PRIMARY OBJECTIVE. The agent earns reward by moving at the
    commanded velocity. This is the main task: "Go fast in the direction I tell you."

    Key insight: We measure velocity in the ROBOT'S LOCAL FRAME (rotated with robot),
    not the world frame. This allows the robot to move in any direction by just
    rotating, then moving forward.

    ROBOT FRAMES:
    ────────────
    World frame: Fixed reference frame (x=North, y=East, z=Up)
    
    Robot frame: Attached to robot's body
      - x-axis: forward direction (along robot's nose)
      - y-axis: left direction (perpendicular to forward)
      - z-axis: upward
    
    When robot rotates, its frame rotates with it. This makes control intuitive:
    "move forward and turn left" = (vx_cmd, vy_cmd, wz_cmd) in robot frame

    VELOCITY MATCHING:
    ─────────────────
    The agent gets reward for matching commanded velocity:
    - Command: [vx_desired, vy_desired] from human
    - State: [vx_actual, vy_actual] from IMU/odometry
    - Reward: high if they match, low if they differ

    EXPONENTIAL KERNEL:
    ──────────────────
    Reward(error) = exp(-error² / std²)
    
    Properties:
    - Maximum reward of 1.0 when error = 0
    - Smooth gradient (good for learning)
    - Decays quickly with error (beyond std, reward ≈ 0)
    - Very differentiable for gradient descent
    
    Example with std=1.0:
    - error=0.0 → reward = exp(0) = 1.00
    - error=0.5 → reward = exp(-0.25) = 0.78
    - error=1.0 → reward = exp(-1.0) = 0.37
    - error=2.0 → reward = exp(-4.0) = 0.02

    PARAMETERS:
    ──────────
    env : ManagerBasedRLEnv
        Environment
    
    std : float
        Standard deviation of the exponential kernel
        Controls how "strict" velocity matching is
        Typical range: 0.5-1.5 m/s
        - std=0.5: strict matching, high reward only near command
        - std=1.5: loose matching, accepts larger errors
    
    command_name : str
        Name of command buffer (e.g., "base_velocity_target")
    
    asset_cfg : SceneEntityCfg
        Robot asset configuration

    RETURN:
    torch.Tensor of shape (num_envs,)
        Reward in range [0.0, 1.0] for each environment

    MATHEMATICAL FORMULATION:
    ────────────────────────
    1. Get robot quaternion (orientation in world frame)
    2. Get robot velocity in world frame [vx_w, vy_w, vz_w]
    3. Rotate velocity to robot frame using inverse quaternion
    4. Extract xy components (ignore z, which is up/down)
    5. Compute error: E = ||[vx_cmd, vy_cmd] - [vx_robot, vy_robot]||²
    6. Apply exponential kernel: reward = exp(-E / std²)

    EXAMPLE WALKTHROUGH:
    ───────────────────
    Scenario: Commanded to move forward 1.0 m/s, no turning
    
    Timestep 1 (bad):
    - Command: [1.0, 0.0] m/s
    - Robot velocity: [0.1, 0.0] m/s (just starting to move)
    - Error: (1.0-0.1)² + (0-0)² = 0.81
    - Reward: exp(-0.81/1.0) = exp(-0.81) = 0.44
    
    Timestep 10 (learning):
    - Command: [1.0, 0.0] m/s
    - Robot velocity: [0.7, 0.0] m/s
    - Error: (1.0-0.7)² = 0.09
    - Reward: exp(-0.09/1.0) = 0.914
    
    Timestep 100 (converged):
    - Command: [1.0, 0.0] m/s
    - Robot velocity: [0.98, 0.0] m/s
    - Error: (1.0-0.98)² = 0.0004
    - Reward: exp(-0.0004/1.0) = 0.9996 ≈ 1.0

    WEIGHT IN CONFIGURATION:
    ──────────────────────
    Typical weight: 1.0 or higher
    - weight=1.0: This is the main objective
    - weight=2.0: Make it VERY important (must track velocity well)
    - Lower weights make other rewards more competitive

    PHYSICAL INTERPRETATION:
    ───────────────────────
    This reward drives forward motion. The agent learns that:
    1. Moving at commanded velocity = good
    2. Commands rotate with the robot (intuitive control)
    3. Error penalizes both undershoot and overshoot
    4. Smooth exponential gradient enables smooth learning
    """
    
    # Get robot asset
    asset = env.scene[asset_cfg.name]
    
    # Get robot's rotation (quaternion) in world frame
    # This tells us which way the robot is facing
    robot_quat = asset.data.root_quat_w
    # Shape: (num_envs, 4) - [x, y, z, w]
    
    # Get robot's linear velocity in world frame [vx_w, vy_w, vz_w]
    robot_vel_w = asset.data.root_lin_vel_w[:, :3]
    # Shape: (num_envs, 3)
    
    # Transform velocity from world frame to robot frame
    # This rotation takes world velocity and expresses it relative to robot
    # After transformation, x=forward, y=left, z=up
    vel_robot = quat_apply_inverse(robot_quat, robot_vel_w)
    # Shape: (num_envs, 3) - velocity in robot frame
    
    # Get commanded velocity [vx_cmd, vy_cmd] from command manager
    # Commands are already in robot frame, so they naturally match this frame
    cmd_vel = env.command_manager.get_command(command_name)[:, :2]
    # Shape: (num_envs, 2) - [forward_cmd, lateral_cmd]
    
    # Compute velocity error in robot xy plane
    # Only compare forward and lateral motion, not vertical
    vel_error_xy = cmd_vel - vel_robot[:, :2]
    # Shape: (num_envs, 2)
    
    # Compute squared error
    lin_vel_error = torch.sum(torch.square(vel_error_xy), dim=1)
    # Shape: (num_envs,)
    # This is: (vx_err)² + (vy_err)²
    
    # Apply exponential kernel reward
    # std is the standard deviation parameter controlling strictness
    reward = torch.exp(-lin_vel_error / (std ** 2))
    # Shape: (num_envs,)
    # Reward is in range [0, 1] with maximum at zero error
    
    return reward


# ═══════════════════════════════════════════════════════════════════════════════
# REWARD FUNCTION 5: TRACK ANGULAR VELOCITY (PRIMARY OBJECTIVE)
# ═══════════════════════════════════════════════════════════════════════════════

def track_ang_vel_z_world_exp(
    env, 
    command_name: str, 
    std: float, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    ═════════════════════════════════════════════════════════════════════════════
    Reward tracking of angular velocity commands (yaw rotation) in world frame
    using exponential kernel.
    ═════════════════════════════════════════════════════════════════════════════

    PURPOSE:
    ───────
    Similar to track_lin_vel but for ROTATION (turning/yaw).
    
    The agent earns reward for turning at the commanded rotational velocity.
    This controls how fast the robot rotates around the vertical (z) axis.

    WHY YAW IN WORLD FRAME:
    ──────────────────────
    Angular velocity is special: rotation around z (up) is meaningful in both
    world frame and robot frame - they're the same!
    
    Reason: The z-axis (vertical) is aligned with gravity, which is universal.
    Unlike linear velocity (where "forward" depends on orientation), rotation
    around z is the same whether measured in world or robot frame.
    
    This simplifies control: turn command directly matches rotation rate.

    COMMAND SEMANTICS:
    ─────────────────
    Command: [0, 0, wz_cmd]
    - First two elements are ignored here (linear velocity, handled by track_lin_vel)
    - Third element is angular velocity command around z-axis
    - Positive = counter-clockwise (when viewed from above)
    - Negative = clockwise

    EXPONENTIAL KERNEL:
    ──────────────────
    Same as linear velocity:
    Reward = exp(-(error²) / std²)
    
    where error = commanded_yaw_rate - actual_yaw_rate

    PARAMETERS:
    ──────────
    env : ManagerBasedRLEnv
        Environment
    
    command_name : str
        Name of command buffer
    
    std : float
        Kernel standard deviation
        Controls strictness of angular velocity matching
        Typical range: 0.2-1.0 rad/s (slower than linear)
    
    asset_cfg : SceneEntityCfg
        Robot asset configuration

    RETURN:
    torch.Tensor of shape (num_envs,)
        Reward in range [0.0, 1.0]

    MATHEMATICAL FORMULATION:
    ────────────────────────
    1. Get robot angular velocity in world frame [wx, wy, wz]
    2. Extract z component (yaw rotation rate)
    3. Get commanded yaw rate from command manager
    4. Compute error: E = (wz_cmd - wz_actual)²
    5. Apply exponential kernel: reward = exp(-E / std²)

    EXAMPLE:
    ──────
    Scenario: Commanded to turn 1.0 rad/s (roughly 57 degrees/second)
    
    Bad execution:
    - Command: wz = 1.0 rad/s
    - Actual: wz = 0.1 rad/s
    - Error: (1.0 - 0.1)² = 0.81
    - Reward: exp(-0.81/0.25) = exp(-3.24) = 0.04 (very low!)
    
    Good execution:
    - Command: wz = 1.0 rad/s
    - Actual: wz = 0.95 rad/s
    - Error: (1.0 - 0.95)² = 0.0025
    - Reward: exp(-0.0025/0.25) = exp(-0.01) = 0.99 (excellent!)

    WEIGHT IN CONFIGURATION:
    ──────────────────────
    Typical weight: 0.5 (secondary to linear velocity)
    - weight=0.5: Turn reasonably well, but linear motion is more important
    - weight=1.0: Equally important as linear motion
    - Lower values tolerate turning errors more

    TUNING GUIDANCE:
    ───────────────
    If robot doesn't turn well:
    - Increase weight
    - Decrease std (more strict)
    
    If robot turns but doesn't move forward:
    - Decrease weight of angular reward
    - Increase weight of linear reward

    REAL-WORLD CONTEXT:
    ──────────────────
    For Go1 quadruped:
    - Max angular velocity: ~3 rad/s (slow trotting)
    - Typical commands: ±1 rad/s range
    - std=0.5 rad/s is reasonable (±1 std covers most commands)
    """
    
    # Get robot asset
    asset = env.scene[asset_cfg.name]
    
    # Get robot angular velocity in world frame [wx, wy, wz]
    # This is the rate of rotation around each axis
    ang_vel_w = asset.data.root_ang_vel_w
    # Shape: (num_envs, 3) - [roll_rate, pitch_rate, yaw_rate]
    
    # Get commanded angular velocity from command manager
    # This is typically [0, 0, wz_cmd] (only yaw is commanded)
    cmd_ang_vel = env.command_manager.get_command(command_name)
    # Shape: (num_envs, 3)
    
    # Extract z component (yaw rate)
    # Index 2 is the z (vertical) rotation component
    ang_vel_error = cmd_ang_vel[:, 2] - ang_vel_w[:, 2]
    # Shape: (num_envs,) - scalar error for each environment
    
    # Compute squared error
    ang_vel_error_sq = torch.square(ang_vel_error)
    # Shape: (num_envs,)
    
    # Apply exponential kernel reward
    # Note: std is passed as parameter to this function
    reward = torch.exp(-ang_vel_error_sq / (std ** 2))
    # Shape: (num_envs,)
    
    return reward


# ═══════════════════════════════════════════════════════════════════════════════
# REWARD FUNCTION 6: JOINT STABILITY PENALTY
# ═══════════════════════════════════════════════════════════════════════════════

def stand_still_joint_deviation_l1(
    env, 
    command_name: str, 
    command_threshold: float = 0.06, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    ═════════════════════════════════════════════════════════════════════════════
    Penalize offsets from the default joint positions when the command is very small.
    ═════════════════════════════════════════════════════════════════════════════

    PURPOSE:
    ───────
    This function CONDITIONALLY PENALIZES joint deviation from default positions,
    but ONLY when the robot should be standing still.

    Intuition: When told to stand still (command ≈ 0), the robot should assume
    a relaxed standing pose with minimal joint deviation. This:
    1. Reduces energy consumption during rest
    2. Looks natural (robot isn't twitching)
    3. Prepares for next movement
    4. Tests balance and stability

    STANDING STILL:
    ──────────────
    A command is "very small" when:
    ||[vx_cmd, vy_cmd]|| < 0.06 m/s

    This threshold (0.06) is chosen because:
    - Smaller than minimum walking speed (~0.1 m/s)
    - Corresponds to ~10% of typical max speed
    - Avoids penalizing the standing phase of crawling gait

    JOINT DEVIATION:
    ───────────────
    Each joint has a "default" position (usually the relaxed pose):
    - Hip rotation: centered
    - Hip flex: slightly bent (natural standing)
    - Knee: slightly bent (shock absorption)

    Deviation = ||actual_joint_angles - default_joint_angles||

    Using L1 norm (sum of absolute values) encourages sparse solutions:
    - Multiple small deviations worse than one larger deviation
    - Encourages robot to stay close to default pose overall

    MATHEMATICAL FORMULATION:
    ────────────────────────
    mask = (||command_xy|| < 0.06)
    # True if standing still, False if moving
    
    base_reward = mdp.joint_deviation_l1(env, asset_cfg)
    # This computes L1 deviation from default joints
    
    final_reward = base_reward * mask
    # Only apply penalty when standing still

    PARAMETERS:
    ──────────
    env : ManagerBasedRLEnv
        Environment
    
    command_name : str
        Name of command buffer
    
    command_threshold : float
        Speed threshold below which robot should stand still (m/s)
        Default 0.06 m/s
        Adjust based on minimum walking speed
    
    asset_cfg : SceneEntityCfg
        Robot asset configuration

    RETURN:
    torch.Tensor of shape (num_envs,)
        Penalty (as positive value, typically used with negative weight)

    EXAMPLE SCENARIO:
    ────────────────
    Standing still condition:
    - Command: [0.02, 0.03] m/s (very small)
    - ||command|| = sqrt(0.0004 + 0.0009) = 0.036 < 0.06 ✓ standing
    
    Joint check:
    - Hip 1 angle: 0.2 rad vs default 0.0 → deviation = 0.2
    - Hip 2 angle: -0.1 rad vs default 0.0 → deviation = 0.1
    - Knee angle: 0.3 rad vs default 0.25 → deviation = 0.05
    - Total L1 = 0.2 + 0.1 + 0.05 = 0.35
    
    Since standing: penalty = 0.35 (when multiplied by negative weight)
    
    Moving condition:
    - Command: [0.3, 0.5] m/s (significant)
    - ||command|| = 0.58 > 0.06 ✗ not standing
    
    Even with joint deviations:
    - Penalty = 0 (not standing, so don't penalize deviation)
    - Robot free to use any joint angles for walking

    HYPERPARAMETER TUNING:
    ────────────────────
    Weight (typically -1.0 to -5.0):
        - Higher magnitude = stronger penalty on standing deviation
        - Increase if robot twitches while standing
        - Decrease if robot struggles to stand stably
    
    command_threshold (default 0.06):
        - Increase if robot should stand still at higher speeds
        - Decrease for more strict standing still behavior
        - Consider minimum walking speed for your robot

    INTERACTION WITH OTHER REWARDS:
    ──────────────────────────────
    This reward only activates when standing still, so:
    - Doesn't interfere with walking rewards
    - Complements velocity tracking (no reward for standing)
    - Encourages relaxed posture during rest phases
    - Natural behavior: stand → walk → stand → walk

    WHY L1 INSTEAD OF L2:
    ───────────────────
    L1 norm (sum of absolute values) has different properties than L2:
    - L1: |a| + |b|
    - L2: sqrt(a² + b²)
    
    For joint control:
    - L1 encourages sparse solutions (use few joints)
    - L2 would encourage balanced deviations
    - L1 is more natural for joint stiffness behavior
    """
    
    # Get command to check if robot should be standing still
    command = env.command_manager.get_command(command_name)
    # Shape: (num_envs, 3) - [vx, vy, wz]
    
    # Extract xy command magnitude
    command_xy = command[:, :2]
    command_mag = torch.norm(command_xy, dim=1)
    # Shape: (num_envs,)
    
    # Check if standing still (command magnitude very small)
    standing_still = command_mag < command_threshold
    # Shape: (num_envs,) - boolean
    
    # Compute base joint deviation penalty using Isaac Lab utility
    # This is the L1 norm of joint angle deviations from defaults
    joint_deviation = mdp.joint_deviation_l1(env, asset_cfg)
    # Shape: (num_envs,) - positive values, higher = more deviation
    
    # Apply mask: only penalize when standing still
    penalty = joint_deviation * standing_still.float()
    # Shape: (num_envs,)
    # Now penalty is nonzero only when standing still
    
    return penalty


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY OF REWARD FUNCTION ROLES
# ═══════════════════════════════════════════════════════════════════════════════

"""
REWARD FUNCTION SUMMARY & TYPICAL WEIGHTS:
═══════════════════════════════════════════

1. track_lin_vel_xy_yaw_frame_exp (WEIGHT: 1.0)
   Role: PRIMARY OBJECTIVE - Move at commanded velocity
   Design: Exponential kernel, smooth learning
   Scale: [0, 1] with 1.0 at zero error

2. track_ang_vel_z_world_exp (WEIGHT: 0.5)
   Role: SECONDARY OBJECTIVE - Turn at commanded rate
   Design: Exponential kernel, less strict than linear
   Scale: [0, 1] with 1.0 at zero error

3. feet_air_time (WEIGHT: 0.5)
   Role: GAIT QUALITY - Encourage lifting feet
   Design: Sum of air time above threshold at foot contact
   Scale: [0, ∞) in seconds, clamped by episode length

4. feet_slide (WEIGHT: -0.5)
   Role: PENALTY - Discourage sliding feet
   Design: Velocity magnitude when in contact
   Scale: [0, ∞) in m/s, weighted negative

5. stand_still_joint_deviation_l1 (WEIGHT: -0.5)
   Role: PENALTY - Reduce energy during standing
   Design: L1 norm of joint deviations, only when standing
   Scale: [0, ∞) in radians, weighted negative

TYPICAL EPISODE REWARD COMPOSITION:
───────────────────────────────────
For successful walking at 1.0 m/s:

Per step (average):
- track_lin_vel: +0.85 (tracking well)
- track_ang_vel: +0.95 (turning well)
- feet_air_time: +0.10 (good gait)
- feet_slide: -0.05 (minor sliding penalty)
- joint_deviation: 0.00 (moving, not standing)

Total per step: 0.85 + 0.48 + 0.05 - 0.05 + 0.00 = 1.33

Per episode (1000 steps): ~1330 reward
This is a STRONG signal for learning!

REWARD BALANCING PHILOSOPHY:
─────────────────────────────
The weights are tuned so that:
1. Primary objectives (velocity tracking) dominate
2. Secondary objectives (gait quality) provide meaningful feedback
3. Penalties (sliding, deviation) prevent undesired behaviors
4. No single reward term can max out the learning signal
5. Trade-offs naturally arise (speed vs stability)

If any reward term weight is too large, the agent will overfit to
that single objective at the expense of overall performance.

═══════════════════════════════════════════════════════════════════════════════
"""
