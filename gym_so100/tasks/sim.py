import numpy as np

from gym_so100.constants import (
    unnormalize_puppet_gripper_joint,
)

from .bimanual import BimanualTask

BOX_POSE = [None]  # to be changed from outside

"""
Environment for simulated robot bi-manual manipulation, with joint position control
Action space:      [left_arm_qpos (5),             # absolute joint position
                    left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                    right_arm_qpos (5),            # absolute joint position
                    right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

Observation space: {"qpos": Concat[ left_arm_qpos (5),          # absolute joint position
                                    left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                    right_arm_qpos (5),         # absolute joint position
                                    right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                    "qvel": Concat[ left_arm_qvel (5),          # absolute joint velocity (rad)
                                    left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                    right_arm_qvel (5),         # absolute joint velocity (rad)
                                    right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                    "images": {"top": (480x640x3)}              # h, w, c, dtype='uint8'
"""


class BimanualSO100Task(BimanualTask):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        a_len = len(action) // 2
        left_arm_action = action[:a_len]
        right_arm_action = action[a_len:]

        # unnormalize gripper action
        left_arm_action[-1] = unnormalize_puppet_gripper_joint(left_arm_action[-1])
        right_arm_action[-1] = unnormalize_puppet_gripper_joint(right_arm_action[-1])

        env_action = np.concatenate([left_arm_action, right_arm_action])
        super().before_step(env_action, physics)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)


class TransferCubeTask(BimanualSO100Task):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            # reset joint position to keyframe 'prepare'
            physics.named.data.qpos[:] = physics.model.key("prepare").qpos[:]
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7:] = BOX_POSE[0]
            # print(f"{BOX_POSE=}")
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


class InsertionTask(BimanualSO100Task):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            # reset joint position to keyframe 'prepare'
            physics.named.data.qpos[:] = physics.model.key("prepare").qpos[:]
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7 * 2 :] = BOX_POSE[0]  # two objects
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    def get_reward(self, physics):
        # return whether peg touches the pin
        all_contact_pairs = self.get_all_contacts(physics)

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
