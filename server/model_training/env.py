import torch

class MineSweeperEnv:
    def __init__(self, width=10, height=10, n_mines=10,
                 penalty_mine=-50, reward_empty=10, penalty_revisit=-5,
                 proximity_reward=2, initial_exploration_bonus=5,
                 periodic_bonus=10, win_reward=100, step_penalty=-20,
                 variant="standard"):
        self.width = width
        self.height = height
        self.n_mines = n_mines
        self.penalty_mine = penalty_mine
        self.reward_empty = reward_empty
        self.penalty_revisit = penalty_revisit
        self.proximity_reward = proximity_reward
        self.initial_exploration_bonus = initial_exploration_bonus
        self.periodic_bonus = periodic_bonus
        self.win_reward = win_reward
        self.step_penalty = step_penalty
        self.variant = variant  # Add a variant attribute
        self.action_space = width * height
        self.observation_space = torch.tensor([width * height])
        self.reset()

    def reset(self):
        self.board = torch.zeros((self.width, self.height), dtype=torch.float32)
        mine_indices = torch.randperm(self.width * self.height)[:self.n_mines]
        self.mine_positions = [(index // self.width, index % self.width) for index in mine_indices]
        for pos in self.mine_positions:
            self.board[pos] = -1
        self.done = False
        self.total_steps = 0
        self.revealed_cells = torch.zeros_like(self.board, dtype=torch.bool)
        self.adjacent_mines_count = self._calculate_adjacent_mines()
        return self._get_observation()

    def _calculate_adjacent_mines(self):
        adjacent_mines = torch.zeros_like(self.board, dtype=torch.float32)
        for x in range(self.width):
            for y in range(self.height):
                if self.board[x, y] == -1:
                    continue
                count = 0
                for i in range(max(0, x-1), min(self.width, x+2)):
                    for j in range(max(0, y-1), min(self.height, y+2)):
                        if self.board[i, j] == -1:
                            count += 1
                adjacent_mines[x, y] = count
        return adjacent_mines

    def _get_observation(self):
        obs = torch.where(self.revealed_cells, self.adjacent_mines_count, torch.tensor(-2.0))
        return obs.flatten()

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, self.done, {}
        
        x, y = divmod(action, self.width)
        reward = 0

        if self.board[x, y] == -1:
            reward = self.penalty_mine
            self.done = True
        elif not self.revealed_cells[x, y]:
            self.revealed_cells[x, y] = True
            reward = self.reward_empty
            reward += self._calculate_proximity_reward(x, y)
            reward += self.adjacent_mines_count[x, y]
            if self.total_steps < 10:
                reward += self.initial_exploration_bonus
            elif self.total_steps % 10 == 0:
                reward += self.periodic_bonus
            if self._check_win_condition():
                reward += self.win_reward
                self.done = True
        else:
            reward = self.penalty_revisit

        self.total_steps += 1
        if self.total_steps > self.width * self.height:
            reward += self.step_penalty
            self.done = True

        if self.variant == "extended_steps":
            reward += 1  # Additional reward for taking more steps

        return self._get_observation(), reward, self.done, {}

    def _calculate_proximity_reward(self, x, y):
        reward = 0
        for i in range(max(0, x-1), min(self.width, x+2)):
            for j in range(max(0, y-1), min(self.height, y+2)):
                if self.board[i, j] == -1:
                    reward += self.proximity_reward
        return reward

    def _check_win_condition(self):
        non_mine_cells = self.width * self.height - self.n_mines
        revealed_cells = self.revealed_cells.sum().item()
        return revealed_cells == non_mine_cells
