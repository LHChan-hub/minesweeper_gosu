import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))

    def reset_noise(self):
        self.weight_epsilon.copy_(torch.randn(self.out_features, self.in_features))
        self.bias_epsilon.copy_(torch.randn(self.out_features))

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return torch.nn.functional.linear(x, weight, bias)

class NoisyDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(NoisyDQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = NoisyLinear(256, 256)
        self.fc3 = NoisyLinear(256, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def reset_noise(self):
        self.fc2.reset_noise()
        self.fc3.reset_noise()

class PrioritizedReplayBuffer:
    def __init__(self, buffer_size, alpha=0.6):
        self.buffer = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
        self.alpha = alpha

    def add(self, experience, error):
        self.buffer.append(experience)
        priority = (abs(error) + 1e-5) ** self.alpha
        self.priorities.append(priority)

    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities, dtype=np.float32)
        probabilities = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        experiences = [self.buffer[idx] for idx in indices]
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        return experiences, weights, indices

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            priority = (abs(error) + 1e-5) ** self.alpha
            self.priorities[idx] = priority

class BootstrappedNoisyDQNAgent:
    def __init__(self, state_dim, action_dim, n_heads=10, gamma=0.99, lr=1e-4, batch_size=256, buffer_size=100000, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_heads = n_heads
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.epsilon = 1.0  # 탐색률 초기값
        self.epsilon_min = 0.01  # 탐색률 최소값
        self.epsilon_decay = 0.99999  # 탐색률 감소율 (더 천천히 감소)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.beta = beta_start

        self.memory = PrioritizedReplayBuffer(buffer_size, alpha)
        self.models = [NoisyDQN(state_dim, action_dim).to(device) for _ in range(n_heads)]
        self.target_models = [NoisyDQN(state_dim, action_dim).to(device) for _ in range(n_heads)]
        self.optimizers = [optim.Adam(model.parameters(), lr=lr) for model in self.models]
        self.loss_fn = nn.MSELoss()

        self.update_target()
        self.frame_idx = 0

    def update_target(self):
        for model, target_model in zip(self.models, self.target_models):
            target_model.load_state_dict(model.state_dict())

    def remember(self, state, action, reward, next_state, done, error):
        self.memory.add((state, action, reward, next_state, done), error)

    def act(self, state, selected_positions):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        state = state.unsqueeze(0).to(device)
        for model in self.models:
            model.reset_noise()
        q_values = [model(state).cpu().detach().numpy().flatten() for model in self.models]
        q_values_mean = np.mean(q_values, axis=0)
        return np.argmax(q_values_mean)

    def replay(self):
        if len(self.memory.buffer) < self.batch_size:
            return

        self.beta = min(1.0, self.beta_start + self.frame_idx * (1.0 - self.beta_start) / self.beta_frames)
        self.frame_idx += 1

        minibatch, weights, indices = self.memory.sample(self.batch_size, self.beta)
        weights = torch.tensor(weights, dtype=torch.float32).to(device)
        errors = np.zeros(self.batch_size, dtype=np.float32)

        for i in range(self.n_heads):
            states = torch.stack([experience[0] for experience in minibatch]).to(device)
            actions = torch.tensor([experience[1] for experience in minibatch], dtype=torch.long).to(device)
            rewards = torch.tensor([experience[2] for experience in minibatch], dtype=torch.float32).to(device)
            next_states = torch.stack([experience[3] for experience in minibatch]).to(device)
            dones = torch.tensor([experience[4] for experience in minibatch], dtype=torch.float32).to(device)

            q_values = self.models[i](states).gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_values = self.target_models[i](next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

            loss = (self.loss_fn(q_values, target_q_values) * weights).mean()
            self.optimizers[i].zero_grad()
            loss.backward()
            self.optimizers[i].step()

            errors += torch.abs(q_values - target_q_values).cpu().data.numpy()

        self.memory.update_priorities(indices, errors)

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self, path):
        state_dicts = [model.state_dict() for model in self.models]
        torch.save(state_dicts, path)

    def load_model(self, path):
        state_dicts = torch.load(path)
        for model, state_dict in zip(self.models, state_dicts):
            model.load_state_dict(state_dict)
        self.update_target()
