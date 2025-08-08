from gymnasium.envs.registration import register

register(
    id='MazeEnv-v0',
    entry_point="maze_env:MazeEnv",
    kwargs={"grid_size": 10}
)