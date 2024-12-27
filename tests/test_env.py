import gymnasium as gym
import pytest
from gymnasium.utils.env_checker import check_env

import gym_so100  # noqa: F401


@pytest.mark.parametrize(
    "env_task, obs_type",
    [
        # ("SO100Insertion-v0", "state"),
        ("SO100Insertion-v0", "pixels"),
        ("SO100Insertion-v0", "pixels_agent_pos"),
        ("SO100EEInsertion-v0", "pixels"),
        ("SO100EEInsertion-v0", "pixels_agent_pos"),
        ("SO100TransferCube-v0", "pixels"),
        ("SO100TransferCube-v0", "pixels_agent_pos"),
        ("SO100EETransferCube-v0", "pixels"),
        ("SO100EETransferCube-v0", "pixels_agent_pos"),
    ],
)
def test_so100_tasks(env_task, obs_type):
    env = gym.make(env_task, obs_type=obs_type)
    check_env(env.unwrapped)
