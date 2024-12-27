from gymnasium.envs.registration import register

from .policy import InsertionPolicy, PickAndTransferPolicy  # noqa: F401

register(
    id="SO100Insertion-v0",
    entry_point="gym_so100.env:SO100Env",
    max_episode_steps=300,
    # Even after seeding, the rendered observations are slightly different,
    # so we set `nondeterministic=True` to pass `check_env` tests
    nondeterministic=True,
    kwargs={"obs_type": "pixels", "task": "insertion"},
)

register(
    id="SO100EEInsertion-v0",
    entry_point="gym_so100.env:SO100Env",
    max_episode_steps=400,
    # Even after seeding, the rendered observations are slightly different,
    # so we set `nondeterministic=True` to pass `check_env` tests
    nondeterministic=True,
    kwargs={"obs_type": "pixels_agent_pos", "task": "end_effector_insertion"},
)

register(
    id="SO100TransferCube-v0",
    entry_point="gym_so100.env:SO100Env",
    max_episode_steps=300,
    # Even after seeding, the rendered observations are slightly different,
    # so we set `nondeterministic=True` to pass `check_env` tests
    nondeterministic=True,
    kwargs={"obs_type": "pixels", "task": "transfer_cube"},
)

register(
    id="SO100EETransferCube-v0",
    entry_point="gym_so100.env:SO100Env",
    max_episode_steps=400,
    # Even after seeding, the rendered observations are slightly different,
    # so we set `nondeterministic=True` to pass `check_env` tests
    nondeterministic=True,
    kwargs={"obs_type": "pixels_agent_pos", "task": "end_effector_transfer_cube"},
)
