# gym-aloha

A gym environment for ALOHA

<img src="http://remicadene.com/assets/gif/aloha_act.gif" width="50%" alt="ACT policy on ALOHA env"/>


## Installation

Create a virtual environment with Python 3.10 and activate it, e.g. with [`miniconda`](https://docs.anaconda.com/free/miniconda/index.html):
```bash
conda create -y -n aloha python=3.10 && conda activate aloha
```

Install gym-aloha:
```bash
pip install gym-aloha
```


## Quickstart

```python
# example.py
import gymnasium as gym
import gym_aloha

env = gym.make("gym_aloha/AlohaInsertion-v0", render_mode="human")
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    image = env.render()

    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

To use this [example](./example.py) with `render_mode="human"`, you should set the environment variable `export MUJOCO_GL=glfw` or simply run
```bash
MUJOCO_GL=glfw python example.py
```


## Contribute

Instead of using `pip` directly, we use `poetry` for development purposes to easily track our dependencies.
If you don't have it already, follow the [instructions](https://python-poetry.org/docs/#installation) to install it.

Install the project with dev dependencies:
```bash
poetry install --with dev
```

### Add dependencies

The equivalent of `pip install some-package` would just be:
```bash
poetry add some-package
```

### Follow our style

```bash
# install pre-commit hooks
pre-commit install

# apply style and linter checks on staged files
pre-commit
```


## Acknowledgment

gym-aloha is adapted from [ALOHA](https://tonyzhaozh.github.io/aloha/)
