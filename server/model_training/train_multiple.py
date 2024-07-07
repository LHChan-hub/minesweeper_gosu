import sys
import os
import importlib
import logging
from itertools import product

current_dir = os.path.dirname(__file__)
gym_minesweeper_dir = os.path.abspath(os.path.join(current_dir, '../../gym_minesweeper'))
sys.path.append(gym_minesweeper_dir)

import gym
import numpy as np
import torch

from gym.envs.registration import register

module_name = "gym_minesweeper"
spec = importlib.util.spec_from_file_location(module_name, os.path.join(gym_minesweeper_dir, "__init__.py"))
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)

register(
    id='Minesweeper-v0',
    entry_point='gym_minesweeper.minesweeper_env:MinesweeperEnv',
)

from dqn_agent import DQNAgent, device

# 로그 파일 설정
logging.basicConfig(filename='training_log.txt', level=logging.INFO, format='%(message)s')

def main():
    env = gym.make('Minesweeper-v0')
    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_dim = env.action_space.n

    # 하이퍼파라미터 조합 설정
    gamma_values = [0.99, 0.95]
    lr_values = [1e-3, 1e-4]
    batch_size_values = [64, 32]
    epsilon_decay_values = [0.995, 0.99]

    param_combinations = list(product(gamma_values, lr_values, batch_size_values, epsilon_decay_values))

    def save_model(agent, param_index):
        model_path = os.path.join(current_dir, 'models', f'saved_model_{param_index}.pth')
        torch.save(agent.model.state_dict(), model_path)
        print(f"Model saved successfully at {model_path}.")

    def epsilon_to_percentage(epsilon):
        return int((1 - epsilon) * 100)

    for param_index, (gamma, lr, batch_size, epsilon_decay) in enumerate(param_combinations):
        print(f"Training with parameters set {param_index}: gamma={gamma}, lr={lr}, batch_size={batch_size}, epsilon_decay={epsilon_decay}")
        agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, gamma=gamma, lr=lr, batch_size=batch_size, epsilon_decay=epsilon_decay)

        for e in range(100):  # 각 조합에 대해 100 에피소드 학습
            state, _ = env.reset()
            state = state.flatten()
            episode_log = [f"Episode {e + 1} with parameters set {param_index}"]
            episode_log.append(f"Initial Epsilon: {epsilon_to_percentage(agent.epsilon)}%")
            for time in range(500):
                action = agent.act(state)
                x, y = divmod(action, env.width)
                episode_log.append(f"Selected cell: ({x}, {y})")
                next_state, reward, done, _ = env.step(action)
                next_state = next_state.flatten()
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    episode_log.append(f"Final score: {env.score}")
                    episode_log.append(f"Epsilon: {epsilon_to_percentage(agent.epsilon)}%")
                    logging.info('\n'.join(episode_log))
                    print(f"episode: {e + 1}/100, score: {env.score}, epsilon: {epsilon_to_percentage(agent.epsilon)}%")
                    break
                if len(agent.memory) > agent.batch_size:
                    agent.replay()

            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
            episode_log.append(f"Updated Epsilon: {epsilon_to_percentage(agent.epsilon)}%")

        save_model(agent, param_index)

if __name__ == "__main__":
    main()
