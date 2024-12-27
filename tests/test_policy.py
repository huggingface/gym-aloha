import gymnasium as gym
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pytest

from gym_so100.constants import SIM_EPISODE_LENGTH
from gym_so100.policy import InsertionPolicy, PickAndTransferPolicy


@pytest.mark.parametrize(
    "task_name, policy_cls",
    [
        ("SO100EETransferCube-v0", PickAndTransferPolicy),
        ("SO100EEInsertion-v0", InsertionPolicy),
    ],
)
def test_policy(task_name, policy_cls, render=False, episode_num=1):
    # test rolling out policy
    inject_noise = False

    # setup the environment
    env = gym.make(task_name, obs_type="pixels_agent_pos")

    for episode_idx in range(episode_num):
        observation, info = env.reset()
        # init episode with first observation and info
        rewards = [0]
        frames = [observation["pixels"]["top"]]
        if render:
            _, axs = plt.subplots(1, 2, figsize=(64, 16))
            mng = plt.get_current_fig_manager()
            # maximize window, only tested on ubuntu
            backend = str(plt.get_backend())
            if backend == "TkAgg":
                mng.resize(*mng.window.maxsize())
            elif backend == "wxAgg":
                mng.frame.Maximize(True)
            elif backend == "QtAgg":
                mng.window.showMaximized()
            top_img = axs[0].imshow(observation["pixels"]["top"])
            angle_img = axs[1].imshow(observation["pixels"]["angle"])
            plt.ion()

        policy = policy_cls(inject_noise)
        for i in range(SIM_EPISODE_LENGTH):
            action = policy(observation)
            # Only rewards are used to evaluate the policy
            observation, reward, _, _, _ = env.step(action)
            rewards.append(reward)
            frames.append(observation["pixels"]["top"])
            if render and (i % 3 == 0):  # only render every 3 steps for speed
                top_img.set_data(observation["pixels"]["top"])
                angle_img.set_data(observation["pixels"]["angle"])
                plt.pause(0.02)
        plt.close()

        # check last 50 frames for success
        episode_return = np.sum(rewards[-50:])
        if episode_return > 10:
            print(f"{episode_idx=} Successful, {episode_return=}")
            imageio.mimsave(f"example_episode_{episode_idx}.mp4", np.stack(frames), fps=50)
        else:
            print(f"{episode_idx=} Failed")
