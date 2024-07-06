from gym.envs.registration import register

register(
    id='Minesweeper-v0',
    entry_point='minesweeper_env:MinesweeperEnv',
)
