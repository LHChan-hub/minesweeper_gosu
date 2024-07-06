from flask import Flask, request, render_template, jsonify
import torch
import torch.nn as nn
import os
import numpy as np
from gym_minesweeper.minesweeper_env import MinesweeperEnv

app = Flask(__name__, template_folder="client/templates", static_folder="client/static")

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

# 로드된 모델 (미리 학습된 모델을 로드)
model = DQN(state_size=100, action_size=100)  # 10x10 보드 기준
model_path = os.path.join('server', 'model_training', 'models', 'saved_model_100.pth')
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Minesweeper 환경 초기화
env = MinesweeperEnv()

@app.route('/')
def index():
    return render_template('index.html', board=env.board.tolist())

@app.route('/reset', methods=['POST'])
def reset():
    global env
    env.reset()
    return jsonify({'board': env.board.tolist()})

@app.route('/move', methods=['POST'])
def move():
    data = request.json
    state = np.array(data['state'])
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    action = torch.argmax(model(state_tensor)[0]).item()

    # 한 번 선택한 위치를 다시 선택하지 않도록 처리
    while state[action] != 0:
        action = (action + 1) % 100

    obs, reward, done, _ = env.step(action)
    return jsonify({'action': action, 'observation': obs.tolist(), 'reward': reward, 'done': done})

if __name__ == '__main__':
    app.run(debug=True)
