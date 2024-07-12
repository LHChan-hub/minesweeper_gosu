import gym
from gym import spaces
import numpy as np
import random

class MinesweeperEnv(gym.Env):
    def __init__(self, width=10, height=10):
        super(MinesweeperEnv, self).__init__()
        self.width = width
        self.height = height
        self.num_mines = int((self.width * self.height) * 0.1)
        self.action_space = spaces.Discrete(self.width * self.height)
        self.observation_space = spaces.Box(low=0, high=8, shape=(self.width * self.height,), dtype=int)
        self.expand_limit = 2  # 확장 횟수 제한
        self.current_expansions = 0
        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.board = np.zeros((self.width, self.height), dtype=int)
        self.visited = np.zeros((self.width, self.height), dtype=bool)
        self.selected_count = np.zeros((self.width, self.height), dtype=int)
        self.num_mines = int((self.width * self.height) * 0.1)
        self._place_mines()
        self._calculate_adjacent_mines()
        self.current_expansions = 0
        return self.board.flatten(), {}

    def _place_mines(self):
        mines = random.sample(range(self.width * self.height), self.num_mines)
        for mine in mines:
            x, y = divmod(mine, self.width)
            self.board[x, y] = -1

    def _calculate_adjacent_mines(self):
        for x in range(self.width):
            for y in range(self.height):
                if self.board[x, y] == -1:
                    continue
                self.board[x, y] = self._count_adjacent_mines(x, y)

    def _count_adjacent_mines(self, x, y):
        count = 0
        for i in range(max(0, x - 1), min(self.width, x + 2)):
            for j in range(max(0, y - 1), min(self.height, y + 2)):
                if self.board[i, j] == -1:
                    count += 1
        return count

    def step(self, action):
        x, y = divmod(action, self.width)
        self.selected_count[x, y] += 1

        if self.visited[x, y]:
            reward = 0
            done = False
            return self.board.flatten(), reward, done, {}

        self.visited[x, y] = True
        if self.board[x, y] == -1:
            reward = 0
            done = True
            return self.board.flatten(), reward, done, {}

        reward = 1
        done = self._check_completion()
        if done:
            if self.current_expansions < self.expand_limit:
                self.expand_board(self.width + 1, self.height + 1)
                self.current_expansions += 1
                reward += 10
            else:
                done = True
        return self.board.flatten(), reward, done, {}

    def _check_completion(self):
        for x in range(self.width):
            for y in range(self.height):
                if not self.visited[x, y] and self.board[x, y] != -1:
                    return False
        return True

    def expand_board(self, new_width, new_height):
        new_board = np.zeros((new_width, new_height), dtype=int)
        new_visited = np.zeros((new_width, new_height), dtype=bool)

        min_width = min(self.width, new_width)
        min_height = min(self.height, new_height)

        new_board[:min_width, :min_height] = self.board[:min_width, :min_height]
        new_visited[:min_width, :min_height] = self.visited[:min_width, :min_height]

        self.width = new_width
        self.height = new_height
        self.board = new_board
        self.visited = new_visited
        self.selected_count = np.zeros((self.width, self.height), dtype=int)
        self.action_space = spaces.Discrete(self.width * self.height)
        self.observation_space = spaces.Box(low=0, high=8, shape=(self.width * self.height,), dtype=int)

        self.num_mines = int((self.width * self.height) * 0.1)
        self._place_mines()
        self._calculate_adjacent_mines()
