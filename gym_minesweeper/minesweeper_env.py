import numpy as np
import gym
from gym import spaces
import torch
import time

class MinesweeperEnv(gym.Env):
    def __init__(self, width=10, height=10, num_mines=10, max_steps=1000, time_penalty=0.5):
        super(MinesweeperEnv, self).__init__()
        self.initial_width = width
        self.initial_height = height
        self.initial_num_mines = num_mines
        self.width = width
        self.height = height
        self.num_mines = num_mines
        self.max_steps = max_steps
        self.time_penalty = time_penalty
        self.board = np.zeros((width, height), dtype=int)
        self.revealed = np.zeros((width, height), dtype=bool)
        self.mines = set()
        self.selected_positions = set()
        self.score = 0
        self.steps = 0
        self.start_time = time.time()
        self.action_space = spaces.Discrete(width * height)
        self.observation_space = spaces.Box(low=-1, high=8, shape=(width, height), dtype=np.float32)
        self.reset()

    def reset(self):
        self.width = self.initial_width
        self.height = self.initial_height
        self.num_mines = self.initial_num_mines
        self.board = np.zeros((self.width, self.height), dtype=int)
        self.revealed = np.zeros((self.width, self.height), dtype=bool)
        self.mines = set()
        self.selected_positions = set()
        self.score = 0
        self.steps = 0
        self.start_time = time.time()
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
        self.steps += 1
        elapsed_time = time.time() - self.start_time

        if self.board[x, y] == -1:
            self.revealed[x, y] = True
            self.score -= 10
            done = True
            if self.steps <= 5:
                # 처음 5번의 선택 중 지뢰를 찾은 경우 학습에서 제외
                return self.get_state(), float(self.score), done, {'early_termination': True}
            return self.get_state(), float(self.score), done, {}  # 지뢰 선택 시 -10점 및 게임 종료

        self.revealed[x, y] = True
        penalty = min(self.time_penalty * elapsed_time, 0.5)  # 패널티는 최대 0.5점까지 감소
        self.score += 1 - penalty

        if self.is_done():
            self.expand_board()

        return self.get_state(), float(self.score), self.is_done(), {}  # 지뢰가 아닌 칸 선택 시 점수 반환

    def expand_board(self):
        self.width += self.initial_width
        self.height += self.initial_height
        self.num_mines += self.initial_num_mines
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
        state = np.where(self.revealed, self.board, -1)
        return torch.tensor(state, dtype=torch.float32)

    def is_done(self):
        return np.all(self.revealed | (self.board == -1))

    def render(self, mode='human', close=False):
        if mode == 'human':
            for y in range(self.height):
                row = ''
                for x in range(self.width):
                    if self.revealed[x, y]:
                        if self.board[x, y] == -1:
                            row += ' * '
                        else:
                            row += f' {self.board[x, y]} '
                    else:
                        row += ' . '
                print(row)
            print()
