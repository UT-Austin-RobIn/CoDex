"""Grasp-move-actuate primitive settings shared by RL training, evaluation and
grasp selection. Stage 3 screens a plain lift; training/evaluation can carry to a goal."""

# Physical parameters drawn once per parallel environment, uniformly from
# (low, high); a single number pins every environment to it. Friction is the
# coefficient of a contact pair: hand/functional object and table/functional object.
# Mass and size are multipliers on the reconstructed mesh (uniform density
# 1000 kg/m^3, registered scale).
DOMAIN_RANDOMIZATION = {
    "hand_friction": (0.4, 1.0),
    "table_friction": (0.2, 0.8),
    "mass_scale": (0.9, 1.1),
    "size_scale": (0.98, 1.02),
}

PRIMITIVE_ENV_CONFIG = {
    **DOMAIN_RANDOMIZATION,
    "max_episode_steps": 1,
    "pregrasp_offset": 0.1,
    "lift_height": 0.3,
    "num_approach_steps": 100,
    "num_grasp_steps": 150,
    "num_lift_steps": 100,
    "approach_step_size": 0.2,
    "grasp_step_size": 0.2,
    "lift_step_size": 0.2,
    "pose_est_noise": 0.01,
    "action_mode": "all",
    "control_mode": "pd_joint_delta_pos",
    "robot_uids": "floating_leap_hand_right",
}

# Actuation phase: controller steps driving the press, and the final window
# over which the selected fingertip must keep pressing.
NUM_ACTUATE_STEPS = 120
PRESS_HOLD_SECONDS = 1.0

# Residual action scales (hand position [m], orientation [rad], finger joints [rad]).
DELTA_POS_SCALE = 0.05
DELTA_ROT_SCALE = 0.1
DELTA_FINGER_SCALE = 0.5
