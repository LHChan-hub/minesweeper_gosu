import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, env, initial_tau=1.0, min_tau=0.01, decay_steps=40000, power=0.1):
        self.env = env
        self.buffer_size = 100000
        self.batch_size = 128
        self.gamma = 0.98
        self.learning_rate = 0.00025
        self.target_update_interval = 1000

        self.replay_buffer = ReplayBuffer(self.buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = QNetwork(env.observation_space.shape[0], env.action_space.n).to(self.device)
        self.target_network = QNetwork(env.observation_space.shape[0], env.action_space.n).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()

        self.scaler = GradScaler()

        self._initialize_weights()

        self.steps = 0
        self.tau = initial_tau
        self.min_tau = min_tau
        self.decay_steps = decay_steps
        self.power = power
        self.visited = set()

    def _initialize_weights(self):
        for m in self.q_network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def select_action(self, state):
        state_tuple = tuple(state)
        if state_tuple in self.visited:
            return None

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state).cpu().numpy().flatten()

        q_values = q_values - np.max(q_values)
        exp_q = np.exp(q_values / self.tau)
        probs = exp_q / np.sum(exp_q)

        if np.any(np.isnan(probs)):
            probs = np.ones_like(probs) / len(probs)

        action = np.random.choice(range(len(probs)), p=probs)

        self.visited.add(state_tuple)

        return action

    def update(self, episode):
        if len(self.replay_buffer) < self.batch_size:
            return None

        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        state = torch.FloatTensor(np.array(state)).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        action = torch.LongTensor(np.array(action)).to(self.device)
        reward = torch.FloatTensor(np.array(reward)).to(self.device)
        done = torch.FloatTensor(np.array(done)).to(self.device)

        with autocast():
            q_values = self.q_network(state)
            next_q_values = self.target_network(next_state)
            q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
            next_q_value = next_q_values.max(1)[0]
            expected_q_value = reward + self.gamma * next_q_value * (1 - done)

            td_errors = expected_q_value.detach() - q_value
            loss = (td_errors ** 2).mean()
            loss = torch.clamp(loss, max=1e2)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.steps += 1
        if self.steps % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        tau_factor = (self.decay_steps - episode) / self.decay_steps
        self.tau = max(self.min_tau, self.tau * tau_factor ** self.power)

        return loss.item()
