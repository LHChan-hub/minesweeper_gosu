import numpy as np
import gym
from gym import spaces
import torch

class MinesweeperEnv(gym.Env):
    def __init__(self, width=10, height=10, num_mines=10):
        super(MinesweeperEnv, self).__init__()
        self.width = width
        self.height = height
        self.num_mines = num_mines
        self.board = np.zeros((width, height), dtype=int)
        self.revealed = np.zeros((width, height), dtype=bool)
        self.mines = set()
        self.selected_positions = set()
        self.score = 0
        self.action_space = spaces.Discrete(width * height)
        self.observation_space = spaces.Box(low=-1, high=8, shape=(width, height), dtype=np.float32)
        self.reset()

    def reset(self):
        self.board.fill(0)
        self.revealed.fill(False)
        self.mines = set()
        self.selected_positions = set()
        self.score = 0
        while len(self.mines) < self.num_mines:
            x, y = np.random.randint(self.width), np.random.randint(self.height)
            self.mines.add((x, y))
        for (x, y) in self.mines:
            self.board[x, y] = -1
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height and self.board[nx, ny] != -1:
                        self.board[nx, ny] += 1
        return self.get_state()

    def step(self, action):
        x, y = divmod(action, self.width)
        if (x, y) in self.selected_positions:
            return self.get_state(), 0.0, False, {}  # 이미 선택된 칸이면 점수 변화 없음

        self.selected_positions.add((x, y))
        if self.board[x, y] == -1:
            self.revealed[x, y] = True
            self.score -= 10
            return self.get_state(), float(self.score), True, {}  # 지뢰 선택 시 -10점 및 게임 종료
        self.revealed[x, y] = True
        self.score += 10
        return self.get_state(), float(self.score), self.is_done(), {}  # 지뢰가 아닌 칸 선택 시 +10점

    def get_state(self):
        state = np.where(self.revealed, self.board, -1)
        return torch.tensor(state, dtype=torch.float32)

    def is_done(self):
        return np.all(self.revealed | (self.board == -1))

    def render(self, mode='human', close=False):
        pass
