import gym
from gym import spaces
import numpy as np
import random

class MinesweeperEnv(gym.Env):
    def __init__(self, rows=10, cols=10, mines=10):
        super(MinesweeperEnv, self).__init__()
        self.rows = rows
        self.cols = cols
        self.mines = mines
        self.action_space = spaces.Discrete(rows * cols)
        self.observation_space = spaces.Box(low=0, high=2, shape=(rows * cols,), dtype=np.int32)
        self.reset()
        self.first_step = True

    def reset(self, seed=None, options=None):
        self.board = np.zeros((self.rows, self.cols), dtype=np.int32)
        self.selected = np.zeros((self.rows, self.cols), dtype=bool)
        self.mine_positions = set()
        self.safe_cells_count = 0
        self.done = False
        self.score = 0
        self.first_step = True
        return self._get_observation(), {}

    def _get_observation(self):
        obs = np.copy(self.board)
        obs[~self.selected] = 0  # 선택되지 않은 셀은 0으로 표시
        return obs.flatten()

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, self.done, {}

        row = action // self.cols
        col = action % self.cols

        if self.selected[row, col]:
            return self._get_observation(), 0, False, {}

        if self.first_step:
            self._place_mines(row, col)
            self.first_step = False

        if self.board[row, col] == -1:
            self.done = True
            reward = -1
        else:
            self._reveal(row, col)
            reward = 1
            if self.board[row, col] == 0:
                self._reveal_safe_areas(row, col)
            self.safe_cells_count += 1
            self.score += reward
            if self.safe_cells_count == (self.rows * self.cols - self.mines):
                self.done = True

        return self._get_observation(), reward, self.done, {}

    def _place_mines(self, initial_row, initial_col):
        while len(self.mine_positions) < self.mines:
            pos = (random.randint(0, self.rows - 1), random.randint(0, self.cols - 1))
            if pos not in self.mine_positions and pos != (initial_row, initial_col):
                self.mine_positions.add(pos)
                self.board[pos] = -1  # -1은 지뢰를 나타냅니다.

        for row in range(self.rows):
            for col in range(self.cols):
                if self.board[row, col] != -1:
                    self.board[row, col] = self._count_mines_around(row, col)

    def _count_mines_around(self, row, col):
        count = 0
        for r in range(max(0, row - 1), min(self.rows, row + 2)):
            for c in range(max(0, col - 1), min(self.cols, col + 2)):
                if self.board[r, c] == -1:
                    count += 1
        return count

    def _reveal(self, row, col):
        if 0 <= row < self.rows and 0 <= col < self.cols:
            if not self.selected[row, col]:
                self.selected[row, col] = True

    def _reveal_safe_areas(self, row, col):
        stack = [(row, col)]
        while stack:
            r, c = stack.pop()
            if self.board[r, c] == 0:
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.rows and 0 <= nc < self.cols and not self.selected[nr, nc]:
                            self._reveal(nr, nc)
                            if self.board[nr, nc] == 0:
                                stack.append((nr, nc))

    def render(self, mode='human'):
        print("Board:")
        for row in range(self.rows):
            for col in range(self.cols):
                if self.selected[row, col]:
                    print(self.board[row, col], end=' ')
                else:
                    print('.', end=' ')
            print()
        print(f"Score: {self.score}")

# 환경 테스트
if __name__ == "__main__":
    env = MinesweeperEnv()
    obs, info = env.reset()
    env.render()
    done = False
    while not done:
        action = env.action_space.sample()  # 무작위 행동 선택
        obs, reward, done, _ = env.step(action)
        env.render()
        print(f"Action: {action}, Reward: {reward}, Done: {done}")
