# app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
import random
import torch

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/click_cell', methods=['POST'])
def click_cell():
    data = request.json
    x = data['x']
    y = data['y']
    # 게임 로직 처리
    result = handle_click(x, y)
    return jsonify(result)

def handle_click(x, y):
    # 셀 클릭 로직
    # 예시로 간단히 구현
    if random.random() < 0.1:
        return {"status": "mine", "x": x, "y": y}
    else:
        return {"status": "safe", "x": x, "y": y}

# 학습된 모델 로드
model = torch.load('path_to_trained_model.pt')

def agent_action(state):
    state = torch.tensor(state, dtype=torch.float32)
    with torch.no_grad():
        action = model(state).argmax().item()
    return action

@app.route('/agent_play', methods=['POST'])
def agent_play():
    state = request.json['state']
    action = agent_action(state)
    return jsonify({"action": action})

if __name__ == '__main__':
    app.run(debug=True)
