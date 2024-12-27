import argparse

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from gym_so100.constants import DT, FPS, JOINTS, SIM_EPISODE_LENGTH
from gym_so100.policy import InsertionPolicy, PickAndTransferPolicy

DEFAULT_FEATURES = {
    "next.reward": {
        "dtype": "float32",
        "shape": (1,),
        "names": None,
    },
    "next.success": {
        "dtype": "bool",
        "shape": (1,),
        "names": None,
    },
    "seed": {
        "dtype": "int64",
        "shape": (1,),
        "names": None,
    },
    "timestamp": {
        "dtype": "float32",
        "shape": (1,),
        "names": None,
    },
}


def record_dataset(num_episodes: int, user_id: str, env_name: str, policy_cls, **kwargs):
    """record pick_and_transfer policy"""

    dataset_dir = kwargs["root"] + "/" + user_id + "/" + env_name
    onscreen_render = kwargs["render"]
    inject_noise = True

    # setup the environment
    env = gym.make(env_name, obs_type="pixels_agent_pos")

    features = DEFAULT_FEATURES
    # image features
    image_keys = env.observation_space["pixels"].keys()
    for key in image_keys:
        features[f"observation.images.{key}"] = {
            "dtype": "video",
            "shape": env.observation_space["pixels"][key].shape,
            "names": ["channel", "height", "width"],
        }
    # states features
    features["observation.state"] = {
        "dtype": "float32",
        "shape": env.observation_space["agent_pos"].shape,
        "names": JOINTS,
    }

    # action feature
    features["action"] = {
        "dtype": "float32",
        "shape": env.observation_space["agent_pos"].shape,
        "names": JOINTS,
    }

    dataset = LeRobotDataset.create(
        user_id,
        FPS,
        root=dataset_dir,
        features=features,
        image_writer_processes=1,
        image_writer_threads=4,
    )

    episode_idx = 0
    while True:
        observation, _ = env.reset()

        # create policy for action
        policy = policy_cls(inject_noise)

        # init rewards info
        rewards = [0]
        if onscreen_render:
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

        for i in range(SIM_EPISODE_LENGTH):
            data_frame = {
                "observation.state": observation["agent_pos"],
                "seed": 0,
                "timestamp": DT * i,
            }
            action = policy(observation)
            observation, reward, _, _, _ = env.step(action)
            data_frame["next.reward"] = reward
            data_frame["next.success"] = reward >= 4 and i == SIM_EPISODE_LENGTH - 1
            data_frame["action"] = observation["agent_pos"]
            for key in image_keys:
                data_frame["observation.images." + key] = observation["pixels"][key]

            dataset.add_frame(data_frame)

            rewards.append(reward)
            if onscreen_render and (i % 3 == 0):  # only render every 3 steps for speed
                top_img.set_data(observation["pixels"]["top"])
                angle_img.set_data(observation["pixels"]["angle"])
                plt.pause(0.02)
        plt.close()

        episode_idx += 1
        # check last 50 frames for success
        episode_return = np.sum(rewards[-50:])
        if episode_return > 0:
            print(f"{episode_idx=} Successful, {episode_return=}")
            print(f"Saving successful task to episode {dataset.num_episodes}")
            dataset.save_episode(task=env_name)
            if dataset.num_episodes == num_episodes:
                break
        else:
            print(f"{episode_idx=} Failed, last saved successful episode count {dataset.num_episodes}")
            dataset.clear_episode_buffer()

    dataset.consolidate()
    print(
        f"Finished recording, tried {episode_idx} times, success {dataset.num_episodes}, success rate: {dataset.num_episodes/episode_idx}"
    )
    print(f"Dataset saved in {dataset_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, default=2, help="Number of episodes to record")
    parser.add_argument("--user-id", type=str, default="test_user", help="User ID for the dataset")
    parser.add_argument("--root", type=str, default="dataset", help="Root dir to save recordings")
    parser.add_argument("--env-name", type=str, default="SO100EETransferCube-v0", help="Environment name")
    parser.add_argument("--render", type=bool, default=False, help="Render the environment on display")

    """
    Example usage:
    python record_lerobot_dataset.py --num_episodes 5 --env_name SO100EETransferCube-v0 --policy PickAndTransferPolicy
    """
    args = parser.parse_args()
    kwargs = vars(args)

    # fill task info
    if args.env_name == "SO100EETransferCube-v0":
        kwargs["policy_cls"] = PickAndTransferPolicy
    elif args.env_name == "SO100EEInsertion-v0":  # TODO: Calib the insertion policy
        kwargs["policy_cls"] = InsertionPolicy
    else:
        raise ValueError(f"Unsupported environment name: {args.env_name}")

    record_dataset(**kwargs)
