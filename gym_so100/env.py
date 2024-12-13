import gymnasium as gym
import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from gymnasium import spaces

from gym_so100.constants import (
    ASSETS_DIR,
    DT,
    JOINTS,
)
from gym_so100.tasks.sim import BOX_POSE, InsertionTask, TransferCubeTask
from gym_so100.tasks.sim_end_effector import (
    InsertionEndEffectorTask,
    TransferCubeEndEffectorTask,
)
from gym_so100.utils import sample_box_pose, sample_insertion_pose


class SO100Env(gym.Env):
    # TODO(aliberts): add "human" render_mode
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        task,
        obs_type="pixels",
        render_mode="rgb_array",
        observation_width=640,
        observation_height=480,
        visualization_width=640,
        visualization_height=480,
    ):
        super().__init__()
        self.task = task
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height

        self._env = self._make_env_task(self.task)

        # check if end effector task
        self.is_ee_task = bool(self.task.startswith("end_effector_"))

        if self.obs_type == "state":
            raise NotImplementedError()
            self.observation_space = spaces.Box(
                low=np.array([0] * len(JOINTS)),  # ???
                high=np.array([255] * len(JOINTS)),  # ???
                dtype=np.float64,
            )
        elif self.obs_type == "pixels":
            self.observation_space = spaces.Dict(
                {
                    "top": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.observation_height, self.observation_width, 3),
                        dtype=np.uint8,
                    ),
                    "angle": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.observation_height, self.observation_width, 3),
                        dtype=np.uint8,
                    ),
                }
            )
        elif self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(
                        {
                            "top": spaces.Box(
                                low=0,
                                high=255,
                                shape=(self.observation_height, self.observation_width, 3),
                                dtype=np.uint8,
                            ),
                            "angle": spaces.Box(
                                low=0,
                                high=255,
                                shape=(self.observation_height, self.observation_width, 3),
                                dtype=np.uint8,
                            ),
                        }
                    ),
                    "agent_pos": spaces.Box(
                        low=-1000.0,
                        high=1000.0,
                        shape=(len(JOINTS),),
                        dtype=np.float64,
                    ),
                }
            )

        # Add mocap_pose for ee tasks
        if self.is_ee_task:
            self.observation_space["mocap_pose_left"] = spaces.Box(
                low=-1000.0, high=1000.0, shape=(7,), dtype=np.float64
            )
            self.observation_space["mocap_pose_right"] = spaces.Box(
                low=-1000.0, high=1000.0, shape=(7,), dtype=np.float64
            )
            env_state_dim = 7 if "transfer_cube" in self.task else 14
            self.observation_space["env_state"] = spaces.Box(
                low=-1000.0, high=1000.0, shape=(env_state_dim,), dtype=np.float64
            )

        # Define the action space
        if self.is_ee_task:
            # 7 pose of arm + 1 DoF for gripper, (7 + 1) * 2 for both arms
            self.action_space = spaces.Box(low=-1, high=1, shape=(16,), dtype=np.float32)
        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(len(JOINTS),), dtype=np.float32)

    def render(self):
        return self._render(visualize=True)

    def _render(self, visualize=False):
        assert self.render_mode == "rgb_array"
        width, height = (
            (self.visualization_width, self.visualization_height)
            if visualize
            else (self.observation_width, self.observation_height)
        )
        # if mode in ["visualize", "human"]:
        #     height, width = self.visualize_height, self.visualize_width
        # elif mode == "rgb_array":
        #     height, width = self.observation_height, self.observation_width
        # else:
        #     raise ValueError(mode)
        # TODO(rcadene): render and visualizer several cameras (e.g. angle, front_close)
        image = self._env.physics.render(height=height, width=width, camera_id="top")
        return image

    def _make_env_task(self, task_name):
        # time limit is controlled by StepCounter in env factory
        time_limit = float("inf")

        if task_name == "transfer_cube":
            xml_path = ASSETS_DIR / "bimanual_transfer_cube.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = TransferCubeTask()
        elif task_name == "insertion":
            xml_path = ASSETS_DIR / "bimanual_insertion.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = InsertionTask()
        elif task_name == "end_effector_transfer_cube":
            xml_path = ASSETS_DIR / "bimanual_end_effector_transfer_cube.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = TransferCubeEndEffectorTask()
        elif task_name == "end_effector_insertion":
            xml_path = ASSETS_DIR / "bimanual_end_effector_insertion.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = InsertionEndEffectorTask()
        else:
            raise NotImplementedError(task_name)

        env = control.Environment(
            physics, task, time_limit, control_timestep=DT, n_sub_steps=None, flat_observation=False
        )
        return env

    def _format_raw_obs(self, raw_obs):
        if self.obs_type == "state":
            raise NotImplementedError()
        elif self.obs_type == "pixels":
            obs = {
                "top": raw_obs["images"]["top"].copy(),
                "angle": raw_obs["images"]["angle"].copy(),
            }
        elif self.obs_type == "pixels_agent_pos":
            obs = {
                "pixels": {
                    "top": raw_obs["images"]["top"].copy(),
                    "angle": raw_obs["images"]["angle"].copy(),
                },
                "agent_pos": raw_obs["qpos"],
            }

        if self.is_ee_task:
            obs["mocap_pose_left"] = raw_obs["mocap_pose_left"].copy()
            obs["mocap_pose_right"] = raw_obs["mocap_pose_right"].copy()
            obs["env_state"] = raw_obs["env_state"].copy()

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # TODO(rcadene): how to seed the env?
        if seed is not None:
            self._env.task.random.seed(seed)
            self._env.task._random = np.random.RandomState(seed)

        # TODO(rcadene): do not use global variable for this
        if "transfer_cube" in self.task:
            BOX_POSE[0] = sample_box_pose(seed)  # used in sim reset
        elif "insertion" in self.task:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose(seed))  # used in sim reset
        else:
            raise ValueError(self.task)

        raw_obs = self._env.reset()

        observation = self._format_raw_obs(raw_obs.observation)

        info = {"is_success": False}
        return observation, info

    def step(self, action):
        assert action.ndim == 1
        # TODO(rcadene): add info["is_success"] and info["success"] ?

        _, reward, _, raw_obs = self._env.step(action)

        # TODO(rcadene): add an enum
        terminated = is_success = reward == 4

        info = {"is_success": is_success}

        observation = self._format_raw_obs(raw_obs)

        truncated = False
        return observation, reward, terminated, truncated, info

    def close(self):
        pass
