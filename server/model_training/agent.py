import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import namedtuple, deque
from torch.amp import GradScaler, autocast

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, fc1_units=128, fc2_units=128, fc3_units=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.output = nn.Linear(fc3_units, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.output(x)

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, state_dim, action_dim, gamma, epsilon, epsilon_decay, epsilon_min, lr, epsilon_reset_threshold=100, epsilon_reset_value=1.0, fc1_units=128, fc2_units=128, fc3_units=128):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = ReplayMemory(50000)
        self.policy_net = DQN(state_dim, action_dim, fc1_units, fc2_units, fc3_units).cuda()
        self.target_net = DQN(state_dim, action_dim, fc1_units, fc2_units, fc3_units).cuda()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.scaler = GradScaler('cuda')
        self.epsilon_reset_threshold = epsilon_reset_threshold
        self.epsilon_reset_value = epsilon_reset_value
        self.update_target()
        self.episode_count = 0
        self.rewards_window = deque(maxlen=100)

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def select_action(self, state):
        sample = random.random()
        if sample > self.epsilon:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float).unsqueeze(0).cuda()
                return self.policy_net(state).max(1)[1].view(1, 1).item()
        else:
            return random.randrange(self.action_dim)

    def store_transition(self, state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)

    def optimize_model(self, batch_size):
        if len(self.memory) < batch_size:
            return

        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.stack([s.clone().detach() for s in batch.state]).cuda()
        action_batch = torch.tensor(batch.action, dtype=torch.long).unsqueeze(1).cuda()
        reward_batch = torch.tensor(batch.reward, dtype=torch.float).cuda()
        non_final_mask = torch.tensor([s is not None for s in batch.next_state], dtype=torch.bool).cuda()
        non_final_next_states = torch.stack([s.clone().detach() for s in batch.next_state if s is not None]).cuda()

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(batch_size).cuda()
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        with autocast(device_type='cuda'):
            loss = self.loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.episode_count += 1
        if self.episode_count % self.epsilon_reset_threshold == 0:
            self.epsilon = self.epsilon_reset_value

    def update_rewards_window(self, reward):
        self.rewards_window.append(reward)
        return sum(self.rewards_window) / len(self.rewards_window)
