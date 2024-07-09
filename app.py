from flask import Flask, render_template, jsonify
import torch
import numpy as np
import os
import sys
import logging

current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir, 'server', 'model_training'))

from dqn_agent import DQNAgent

app = Flask(__name__, template_folder='client/templates', static_folder='client/static')

model_path = os.path.join(current_dir, 'server', 'model_training', 'models', 'saved_model.pth')

# 로그 설정
logging.basicConfig(level=logging.DEBUG)

# 에이전트 설정
state_dim = 100  # 예시로 설정 (10x10 지뢰찾기 보드)
action_dim = 100
agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)

# 환경을 직접 초기화하고 상태를 관리
class SimpleMinesweeper:
    def __init__(self, width=10, height=10, num_mines=10):
        self.width = width
        self.height = height
        self.num_mines = num_mines
        self.board = np.zeros((width, height), dtype=int)
        self.revealed = np.zeros((width, height), dtype=bool)
        self.mines = set()
        self.selected_positions = set()
        self.score = 0
        self.reset()

    def reset(self):
        self.board.fill(0)
        self.revealed.fill(False)
        self.mines = set()
        self.selected_positions = set()
        self.score = 0
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
        return self.get_state(), self.board.tolist()

    def step(self, action):
        x, y = divmod(action, self.width)
        if (x, y) in self.selected_positions:
            return self.get_state(), 0, False, {}  # 이미 선택된 위치면 보상 0

        self.selected_positions.add((x, y))
        if self.board[x, y] == -1:
            self.revealed[x, y] = True
            return self.get_state(), -10, True, {}  # 지뢰 선택 시 -10점 및 게임 종료

        self.revealed[x, y] = True
        self.score += 10
        return self.get_state(), 10, self.is_done(), {}  # 지뢰가 아닌 칸 선택 시 +10점

    def get_state(self):
        state = np.where(self.revealed, self.board, -1)
        return state

    def is_done(self):
        return np.all(self.revealed | (self.board == -1))

    def render(self, mode='human', close=False):
        pass

env = SimpleMinesweeper()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/reset')
def reset():
    state, solution = env.reset()
    return jsonify({"status": "reset successful", "solution": solution})

@app.route('/predict')
def predict():
    if os.path.exists(model_path):
        agent.load_model(model_path)
        logging.info(f"Model loaded successfully from {model_path}")
        state, solution = env.get_state(), env.board.tolist()
        done = False
        predictions = []
        last_action = None
        num_actions = 0

        while not done:
            action = agent.act(state.flatten(), env.selected_positions)  # 선택된 위치 전달
            last_action = action
            x, y = divmod(action, env.width)
            env.selected_positions.add((x, y))  # 위치 추가
            next_state, reward, done, _ = env.step(action)
            logging.debug(f"Action taken: {action} -> (x: {x}, y: {y}), Reward: {reward}, Done: {done}")
            state = next_state
            predictions.append({
                'x': int(x),
                'y': int(y),
                'reward': float(reward),
                'done': bool(done),
                'state': state.tolist()  # 상태 정보를 추가하여 확인
            })
            num_actions += 1
            if done:
                break

        return jsonify(predictions=predictions, solution=solution, last_action=last_action, num_actions=num_actions)
    else:
        return jsonify({"error": "Model not found."})

if __name__ == '__main__':
    app.run(debug=True)
