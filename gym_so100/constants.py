from pathlib import Path

### Simulation envs fixed constants
DT = 0.02  # 0.02 ms -> 1/0.02 = 50 hz
FPS = int(1 / DT)  # frames per second
SIM_EPISODE_LENGTH = 400  # number of steps per episode


JOINTS = [
    # absolute joint position
    "left_arm_rotation",
    "left_arm_pitch",
    "left_arm_elbow",
    "left_arm_wrist_pitch",
    "left_arm_wrist_roll",
    # normalized gripper position 0: close, 1: open
    "left_arm_jaw",
    # absolute joint position
    "right_arm_rotation",
    "right_arm_pitch",
    "right_arm_elbow",
    "right_arm_wrist_pitch",
    "right_arm_wrist_roll",
    # normalized gripper position 0: close, 1: open
    "right_arm_jaw",
]

JOINTS_NUM = len(JOINTS)

ASSETS_DIR = Path(__file__).parent.resolve() / "assets"  # note: absolute path

# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits
MASTER_GRIPPER_JOINT_OPEN = 0.3083
MASTER_GRIPPER_JOINT_CLOSE = -0.6842
PUPPET_GRIPPER_JOINT_OPEN = 0.8
PUPPET_GRIPPER_JOINT_CLOSE = -0.2


############################ Helper functions ############################


def normalize_master_gripper_position(x):
    return (x - MASTER_GRIPPER_POSITION_CLOSE) / (
        MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE
    )


def normalize_puppet_gripper_position(x):
    return (x - PUPPET_GRIPPER_POSITION_CLOSE) / (
        PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE
    )


def unnormalize_master_gripper_position(x):
    return x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE


def unnormalize_puppet_gripper_position(x):
    return x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE


def convert_position_from_master_to_puppet(x):
    return unnormalize_puppet_gripper_position(normalize_master_gripper_position(x))


def normalizer_master_gripper_joint(x):
    return (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)


def normalize_puppet_gripper_joint(x):
    return (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)


def unnormalize_master_gripper_joint(x):
    return x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE


def unnormalize_puppet_gripper_joint(x):
    return x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE


def convert_join_from_master_to_puppet(x):
    return unnormalize_puppet_gripper_joint(normalizer_master_gripper_joint(x))


def normalize_master_gripper_velocity(x):
    return x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)


def normalize_puppet_gripper_velocity(x):
    return x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)


def convert_master_from_position_to_joint(x):
    return (
        normalize_master_gripper_position(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
        + MASTER_GRIPPER_JOINT_CLOSE
    )


def convert_master_from_joint_to_position(x):
    return unnormalize_master_gripper_position(
        (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
    )


def convert_puppet_from_position_to_join(x):
    return (
        normalize_puppet_gripper_position(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
        + PUPPET_GRIPPER_JOINT_CLOSE
    )


def convert_puppet_from_joint_to_position(x):
    return unnormalize_puppet_gripper_position(
        (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
    )
