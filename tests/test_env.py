import gymnasium as gym
import pytest
from gymnasium.utils.env_checker import check_env

import gym_aloha  # noqa: F401


@pytest.mark.parametrize(
    "env_task, obs_type",
    [
        # ("AlohaInsertion-v0", "state"),
        ("AlohaInsertion-v0", "pixels"),
        ("AlohaInsertion-v0", "pixels_agent_pos"),
        ("AlohaTransferCube-v0", "pixels"),
        ("AlohaTransferCube-v0", "pixels_agent_pos"),
    ],
)
def test_aloha(env_task, obs_type):
    env = gym.make(f"gym_aloha/{env_task}", obs_type=obs_type)
    check_env(env.unwrapped)
