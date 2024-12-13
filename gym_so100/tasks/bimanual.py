import collections

import numpy as np
from dm_control.mujoco.engine import Physics
from dm_control.suite import base

from gym_so100.constants import (
    JOINTS_NUM,
    normalize_puppet_gripper_position,
    normalize_puppet_gripper_velocity,
)

BOX_POSE = [None]  # to be changed from outside

"""
Environment for simulated robot bi-manual manipulation

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


class BimanualTask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics: Physics):
        """Pre process action before simulation step"""
        super().before_step(action, physics)

    def initialize_robots(self, physics: Physics):
        """Sets the state of the robots at the start of each task."""
        raise NotImplementedError

    def initialize_episode(self, physics: Physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics: Physics):
        # Get raw qpos: [ left joints(6), right joints(6), cube pose(7) ]
        qpos_raw = physics.data.qpos.copy()
        # 5 DOF arm + 1 DOF gripper
        left_arm_qpos = qpos_raw[:6]
        right_arm_qpos = qpos_raw[6:12]
        # Normalize gripper position to [-1, 1]
        left_arm_qpos[-1] = normalize_puppet_gripper_position(qpos_raw[5])
        right_arm_qpos[-1] = normalize_puppet_gripper_position(qpos_raw[11])
        return np.concatenate([left_arm_qpos, right_arm_qpos])

    @staticmethod
    def get_qvel(physics: Physics):
        qvel_raw = physics.data.qvel.copy()
        left_arm_qvel = qvel_raw[:6]
        right_arm_qvel = qvel_raw[6:12]
        # Normalize gripper velocity to [-1, 1]
        left_arm_qvel[-1] = normalize_puppet_gripper_velocity(qvel_raw[5])
        right_arm_qvel[-1] = normalize_puppet_gripper_velocity(qvel_raw[11])
        return np.concatenate([left_arm_qvel, right_arm_qvel])

    @staticmethod
    def get_env_state(physics: Physics):
        env_state = physics.data.qpos.copy()[JOINTS_NUM:]
        return env_state

    def get_observation(self, physics: Physics):
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos(physics)
        obs["qvel"] = self.get_qvel(physics)
        obs["env_state"] = self.get_env_state(physics)
        obs["images"] = {}
        obs["images"]["top"] = physics.render(height=480, width=640, camera_id="top")
        obs["images"]["angle"] = physics.render(height=480, width=640, camera_id="angle")
        # obs["images"]["vis"] = physics.render(height=480, width=640, camera_id="front_close")

        return obs

    def get_reward(self, physics: Physics):
        # return whether left gripper is holding the box
        raise NotImplementedError

    def get_all_contacts(self, physics: Physics) -> list:
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            name_geom_1 = physics.model.id2name(physics.data.contact[i_contact].geom1, "geom")
            name_geom_2 = physics.model.id2name(physics.data.contact[i_contact].geom2, "geom")
            all_contact_pairs.append((name_geom_1, name_geom_2))

        return all_contact_pairs


class BimanualEETask(BimanualTask):
    def __init__(self, random=None):
        super().__init__(random=random)

    # EE task needs to override this for mocap
    def before_step(self, action, physics: Physics):
        raise NotImplementedError

    def get_observation(self, physics: Physics):
        obs = super().get_observation(physics)

        # used in scripted policy to obtain starting pose
        obs["mocap_pose_left"] = np.concatenate(
            [physics.data.mocap_pos[0], physics.data.mocap_quat[0]]
        ).copy()
        obs["mocap_pose_right"] = np.concatenate(
            [physics.data.mocap_pos[1], physics.data.mocap_quat[1]]
        ).copy()

        # used when replaying joint trajectory
        obs["gripper_ctrl"] = physics.data.ctrl.copy()
        return obs
