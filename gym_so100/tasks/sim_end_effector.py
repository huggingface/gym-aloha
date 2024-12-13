import numpy as np

from gym_so100.constants import (
    JOINTS_NUM,
    PUPPET_GRIPPER_POSITION_CLOSE,
    unnormalize_puppet_gripper_joint,
)
from gym_so100.utils import sample_box_pose, sample_insertion_pose

from .bimanual import BimanualEETask

"""
Environment for simulated robot bi-manual manipulation, with end-effector control.
Action space:      [left_arm_pose (7),             # position and quaternion for end effector
                    left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                    right_arm_pose (7),            # position and quaternion for end effector
                    right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

Observation space: {"qpos": Concat[ left_arm_qpos (6),          # absolute joint position
                                    left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                    right_arm_qpos (6),         # absolute joint position
                                    right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                    "qvel": Concat[ left_arm_qvel (6),          # absolute joint velocity (rad)
                                    left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                    right_arm_qvel (6),         # absolute joint velocity (rad)
                                    right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                    "images": {"main": (480x640x3)}             # h, w, c, dtype='uint8'
"""


class BimanualSO100EndEffectorTask(BimanualEETask):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        a_len = len(action) // 2
        action_left = action[:a_len]
        action_right = action[a_len:]

        # set mocap position and quat to move end-effector
        # left
        np.copyto(physics.data.mocap_pos[0], action_left[:3])
        np.copyto(physics.data.mocap_quat[0], action_left[3:7])
        # right
        np.copyto(physics.data.mocap_pos[1], action_right[:3])
        np.copyto(physics.data.mocap_quat[1], action_right[3:7])

        # set gripper
        g_left_ctrl = unnormalize_puppet_gripper_joint(action_left[7])
        g_right_ctrl = unnormalize_puppet_gripper_joint(action_right[7])
        np.copyto(physics.data.ctrl, np.array([g_left_ctrl, g_right_ctrl]))

    def initialize_robots(self, physics):
        # reset joint position to keyframe 'prepare'
        physics.named.data.qpos[:] = physics.model.key("prepare").qpos[:]

        # update mocap to align with end effector TODO: use precise position of end effector
        physics.forward()
        # left
        left_ee = physics.model.body("so_arm100_left/Fixed_Jaw")
        np.copyto(physics.data.mocap_pos[0], physics.data.xpos[left_ee.id])
        # np.copyto(physics.data.mocap_quat[0], physics.data.xquat[left_ee.id])
        # right
        right_ee = physics.model.body("so_arm100_right/Fixed_Jaw")
        np.copyto(physics.data.mocap_pos[1], physics.data.xpos[right_ee.id])
        # np.copyto(physics.data.mocap_quat[1], physics.data.xquat[right_ee.id])

        # reset gripper control
        close_gripper_control = np.array([PUPPET_GRIPPER_POSITION_CLOSE, PUPPET_GRIPPER_POSITION_CLOSE])
        np.copyto(physics.data.ctrl, close_gripper_control)

    def get_reward(self, physics):
        raise NotImplementedError


class TransferCubeEndEffectorTask(BimanualSO100EndEffectorTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize box position
        cube_pose = sample_box_pose()
        box = physics.model.joint("red_box_joint")
        np.copyto(physics.data.qpos[box.id : box.id + 7], cube_pose)
        # print(f"randomized cube position to {cube_position}")

        super().initialize_episode(physics)

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        all_contact_pairs = self.get_all_contacts(physics)

        touch_left_gripper = ("so_arm100_left/moving_jaw_pad_1", "red_box") in all_contact_pairs
        touch_right_gripper = ("so_arm100_right/moving_jaw_pad_1", "red_box") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs

        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table:  # lifted
            reward = 2
        if touch_left_gripper:  # attempted transfer
            reward = 3
        if touch_left_gripper and not touch_table:  # successful transfer
            reward = 4
        return reward


class InsertionEndEffectorTask(BimanualSO100EndEffectorTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize peg and socket position
        peg_pose, socket_pose = sample_insertion_pose()

        def id2index(j_id: int):
            return JOINTS_NUM + (j_id - JOINTS_NUM) * 7  # first 12 is robot qpos, 7 is pose dim

        peg = physics.model.joint("red_peg_joint")
        peg_start_idx = id2index(peg.id)
        np.copyto(physics.data.qpos[peg_start_idx : peg_start_idx + 7], peg_pose)
        # print(f"randomized cube position to {cube_position}")

        socket = physics.model.joint("blue_socket_joint")
        socket_start_idx = id2index(socket.id)
        np.copyto(physics.data.qpos[socket_start_idx : socket_start_idx + 7], socket_pose)
        # print(f"randomized cube position to {cube_position}")

        super().initialize_episode(physics)

    # TODO: Update rewards for SO-ARM100 robot
    def get_reward(self, physics):
        # return whether peg touches the pin
        all_contact_pairs = all_contact_pairs = self.get_all_contacts(physics)

        touch_right_gripper = ("so_arm100_right/moving_jaw_pad_1", "red_peg") in all_contact_pairs
        touch_left_gripper = (
            ("so_arm100_left/moving_jaw_pad_1", "socket-1") in all_contact_pairs
            or ("so_arm100_left/moving_jaw_pad_1", "socket-2") in all_contact_pairs
            or ("so_arm100_left/moving_jaw_pad_1", "socket-3") in all_contact_pairs
            or ("so_arm100_left/moving_jaw_pad_1", "socket-4") in all_contact_pairs
        )

        peg_touch_table = ("red_peg", "table") in all_contact_pairs
        socket_touch_table = (
            ("socket-1", "table") in all_contact_pairs
            or ("socket-2", "table") in all_contact_pairs
            or ("socket-3", "table") in all_contact_pairs
            or ("socket-4", "table") in all_contact_pairs
        )
        peg_touch_socket = (
            ("red_peg", "socket-1") in all_contact_pairs
            or ("red_peg", "socket-2") in all_contact_pairs
            or ("red_peg", "socket-3") in all_contact_pairs
            or ("red_peg", "socket-4") in all_contact_pairs
        )
        pin_touched = ("red_peg", "pin") in all_contact_pairs

        reward = 0
        if touch_left_gripper and touch_right_gripper:  # touch both
            reward = 1
        if (
            touch_left_gripper and touch_right_gripper and (not peg_touch_table) and (not socket_touch_table)
        ):  # grasp both
            reward = 2
        if peg_touch_socket and (not peg_touch_table) and (not socket_touch_table):  # peg and socket touching
            reward = 3
        if pin_touched:  # successful insertion
            reward = 4
        return reward
