import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95            # 할인 계수
        self.epsilon = 1.0           # 탐험 초기값
        self.epsilon_min = 0.01      # 탐험 최소값
        self.epsilon_decay = 0.995   # 탐험 감소율
        self.learning_rate = 0.001   # 학습률
        self.batch_size = 32         # 배치 크기
        self.model = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        valid_actions = [i for i in range(self.action_size) if state[i] == 0]
        if len(valid_actions) == 0:
            return random.randrange(self.action_size)  # 모든 셀이 선택된 경우 무작위 선택
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        act_values = self.model(state)
        act_values = act_values.cpu().detach().numpy().flatten()
        valid_act_values = {action: act_values[action] for action in valid_actions}
        return max(valid_act_values, key=valid_act_values.get)

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
                target = reward + self.gamma * torch.max(self.model(next_state)).item()
            target_f = self.model(torch.FloatTensor(state).unsqueeze(0).to(device)).detach()
            target_f = target_f.clone()
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(torch.FloatTensor(state).unsqueeze(0).to(device)), target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name, map_location=device))

    def save(self, name):
        torch.save(self.model.state_dict(), name)
