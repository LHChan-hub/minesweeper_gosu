import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NoisyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(NoisyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def add(self, error, transition):
        priority = (abs(error) + 1e-5) ** self.alpha
        self.buffer.append(transition)
        self.priorities.append(priority)

    def sample(self, batch_size, beta):
        priorities = np.array(self.priorities)
        probabilities = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights = weights / weights.max()
        batch = list(zip(*samples))
        states = torch.tensor(np.array(batch[0]), dtype=torch.float32).to(device)
        actions = torch.tensor(batch[1], dtype=torch.int64).to(device)
        rewards = torch.tensor(batch[2], dtype=torch.float32).to(device)
        next_states = torch.tensor(np.array(batch[3]), dtype=torch.float32).to(device)
        dones = torch.tensor(batch[4], dtype=torch.float32).to(device)
        return states, actions, rewards, next_states, dones, indices, torch.tensor(weights, dtype=torch.float32).to(device)

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (abs(error) + 1e-5) ** self.alpha

    def __len__(self):
        return len(self.buffer)

class BootstrappedNoisyDQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, batch_size=32, buffer_size=10000, alpha=0.6, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, hidden_dim=256):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = PrioritizedReplayBuffer(buffer_size, alpha)
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.hidden_dim = hidden_dim
        self.models = [NoisyNetwork(state_dim, action_dim, hidden_dim).to(device)]
        self.target_models = [NoisyNetwork(state_dim, action_dim, hidden_dim).to(device)]
        self.optimizers = [optim.Adam(model.parameters(), lr=0.001) for model in self.models]
        self.loss_fn = nn.SmoothL1Loss()

    def act(self, state, selected_positions):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state = state.to(device)
            q_values = self.models[0](state)
            q_values = q_values.cpu().detach().numpy()
            valid_indices = [pos[0] * int(np.sqrt(self.state_dim)) + pos[1] for pos in selected_positions if pos[0] * int(np.sqrt(self.state_dim)) + pos[1] < self.action_dim]
            q_values[valid_indices] = -np.inf
            return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done, error):
        self.memory.add(error, (state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size, beta=0.4)
        for model, target_model, optimizer in zip(self.models, self.target_models, self.optimizers):
            q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_values = target_model(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            loss = (weights * self.loss_fn(q_values, target_q_values)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            errors = torch.abs(q_values - target_q_values).detach().cpu().numpy()
            self.memory.update_priorities(indices, errors)

        self.epsilon = max(self.epsilon_end, self.epsilon_decay * self.epsilon)

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.models[0].state_dict(),
            'target_model_state_dict': self.target_models[0].state_dict(),
            'optimizer_state_dict': self.optimizers[0].state_dict()
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.models[0].load_state_dict(checkpoint['model_state_dict'])
        self.target_models[0].load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizers[0].load_state_dict(checkpoint['optimizer_state_dict'])

    def update_state_dim(self, new_state_dim):
        self.state_dim = new_state_dim
        self.models = [NoisyNetwork(new_state_dim, self.action_dim, self.hidden_dim).to(device)]
        self.target_models = [NoisyNetwork(new_state_dim, self.action_dim, self.hidden_dim).to(device)]
        self.optimizers = [optim.Adam(model.parameters(), lr=0.001) for model in self.models]
