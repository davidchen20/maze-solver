from gymnasium.envs.registration import register

register(
    id='MazeEnv-walls',
    entry_point="maze_env:MazeEnv",
    kwargs={"grid_size": 10}
)