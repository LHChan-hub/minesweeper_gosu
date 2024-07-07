import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.95, lr=1e-4, batch_size=64, buffer_size=10000, epsilon_decay=0.9999, epsilon_min=0.00001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.epsilon = 1.0  # 탐색률 초기값
        self.epsilon_min = epsilon_min  # 탐색률 최소값
        self.epsilon_decay = epsilon_decay  # 탐색률 감소율

        self.memory = deque(maxlen=buffer_size)
        self.model = DQN(state_dim, action_dim).to(device)
        self.target_model = DQN(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.update_target()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, selected_positions):
        if np.random.rand() <= self.epsilon:
            while True:
                action = random.randrange(self.action_dim)
                if action not in selected_positions:
                    return action
        state = state.unsqueeze(0).to(device)
        q_values = self.model(state)
        action = torch.argmax(q_values).item()
        return action

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.stack([i[0] for i in minibatch]).to(device)
        actions = torch.tensor([i[1] for i in minibatch], dtype=torch.long).to(device)
        rewards = torch.tensor([i[2] for i in minibatch], dtype=torch.float32).to(device)
        next_states = torch.stack([i[3] for i in minibatch]).to(device)
        dones = torch.tensor([i[4] for i in minibatch], dtype=torch.float32).to(device)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.update_target()
