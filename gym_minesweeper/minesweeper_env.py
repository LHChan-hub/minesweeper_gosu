import gym
import numpy as np
from gym import spaces

class MinesweeperEnv(gym.Env):
    def __init__(self, width=10, height=10, num_mines=10, max_steps=1000):
        super(MinesweeperEnv, self).__init__()
        self.initial_width = width
        self.initial_height = height
        self.initial_num_mines = num_mines
        self.width = width
        self.height = height
        self.num_mines = num_mines
        self.max_steps = max_steps
        self.board = np.zeros((width, height), dtype=int)
        self.revealed = np.zeros((width, height), dtype=bool)
        self.mines = set()
        self.selected_positions = set()
        self.score = 0
        self.steps = 0
        self.first_click = True
        self.action_space = spaces.Discrete(width * height)
        self.observation_space = spaces.Box(low=-1, high=8, shape=(width * height,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.width = self.initial_width
        self.height = self.initial_height
        self.num_mines = self.initial_num_mines
        self.board = np.zeros((self.width, self.height), dtype=int)
        self.revealed = np.zeros((self.width, self.height), dtype=bool)
        self.mines = set()
        self.selected_positions = set()
        self.score = 0
        self.steps = 0
        self.first_click = True
        self.place_mines(-1, -1)
        return self.get_state(), {}

    def step(self, action):
        x, y = self.decode_action(action)
        reward = 0
        terminated = False
        if self.board[x, y] == -1:
            self.revealed[x, y] = True
            reward -= 10
            terminated = True
        elif not self.revealed[x, y]:
            self.revealed[x, y] = True
            reward += 1
            if np.sum(self.revealed) == self.width * self.height - self.num_mines:
                self.expand_board()
                reward += 10
        return self.get_state(), reward, terminated, False, {}

    def decode_action(self, action):
        return divmod(action, self.width)

    def place_mines(self, safe_x, safe_y):
        while len(self.mines) < self.num_mines:
            x, y = np.random.randint(self.width), np.random.randint(self.height)
            if (x, y) not in self.mines and not (abs(x - safe_x) <= 1 and abs(y - safe_y) <= 1):
                self.mines.add((x, y))
        for (x, y) in self.mines:
            self.board[x, y] = -1
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height and self.board[nx, ny] != -1:
                        self.board[nx, ny] += 1

    def expand_board(self):
        self.width += 1
        self.height += 1
        self.num_mines += 1
        new_board = np.zeros((self.width, self.height), dtype=int)
        new_revealed = np.zeros((self.width, self.height), dtype=bool)
        new_board[:self.board.shape[0], :self.board.shape[1]] = self.board
        new_revealed[:self.revealed.shape[0], :self.revealed.shape[1]] = self.revealed
        self.board = new_board
        self.revealed = new_revealed
        self.selected_positions = set()
        while len(self.mines) < self.num_mines:
            x, y = np.random.randint(self.width), np.random.randint(self.height)
            if (x, y) not in self.mines:
                self.mines.add((x, y))
        for (x, y) in self.mines:
            self.board[x, y] = -1
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height and self.board[nx, ny] != -1:
                        self.board[nx, ny] += 1

    def get_state(self):
        state = np.where(self.revealed, self.board, -1).astype(np.float32).flatten()
        return state
