import sys
import os
import argparse
import glob
import matplotlib
matplotlib.use('Agg')  # 여기서 백엔드를 Agg로 설정합니다.
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle

current_dir = os.path.dirname(__file__)
gym_minesweeper_dir = os.path.abspath(os.path.join(current_dir, '../../gym_minesweeper'))
sys.path.append(gym_minesweeper_dir)

import gym
import torch

from gym.envs.registration import register
register(
    id='Minesweeper-v0',
    entry_point='minesweeper_env:MinesweeperEnv',
)

from dqn_agent import BootstrappedNoisyDQNAgent, device

def main(args):
    envs = [gym.make('Minesweeper-v0') for _ in range(4)]
    state_size = envs[0].observation_space.shape[0] * envs[0].observation_space.shape[1]
    action_size = envs[0].action_space.n
    agent = BootstrappedNoisyDQNAgent(state_dim=state_size, action_dim=action_size)

    if args.model_file:
        model_path = os.path.join(current_dir, 'models', args.model_file)
        if os.path.exists(model_path):
            agent.load_model(model_path)
            print(f"Model {args.model_file} loaded successfully.")
        else:
            print(f"No model found at {model_path}. Starting training from scratch.")
    else:
        print("No model file specified. Starting training from scratch.")

    def save_model(agent, temp=False):
        model_save_path = os.path.join(current_dir, 'models', 'saved_model.pth')
        agent.save_model(model_save_path)
        print(f"Model saved successfully at {model_save_path}.")
        if not temp:
            model_files = sorted(glob.glob(os.path.join(current_dir, 'models', 'saved_model_*.pth')))
            if len(model_files) > 50:
                os.remove(model_files[0])
                print(f"Removed old model: {model_files[0]}")

    # 학습 기록을 위한 리스트 초기화
    rewards = []
    epsilons = []

    def run_episode(env, e):
        state = env.reset().flatten().to(device)
        selected_positions = set()
        total_reward = 0
        for time in range(500):
            action = agent.act(state, selected_positions)
            x, y = divmod(action, int(env.width))
            selected_positions.add((x, y))
            next_state, reward, done, info = env.step(action)
            next_state = next_state.flatten().to(device)
            if not info.get('early_termination', False):
                error = reward - agent.models[0](state.unsqueeze(0).to(device)).max(1)[0].item()
                agent.remember(state, action, reward, next_state, done, error)
            state = next_state
            total_reward += reward
            if done:
                epsilon_percent = 100 - ((agent.epsilon - agent.epsilon_min) / (1.0 - agent.epsilon_min) * 100)
                print(f"episode: {e + 1}/100000, score: {reward}, epsilon: {epsilon_percent:.2f}%")
                break
            if len(agent.memory.buffer) > agent.batch_size:
                agent.replay()
        return total_reward, agent.epsilon

    with ThreadPoolExecutor(max_workers=4) as executor:
        for e, env in zip(range(100000), cycle(envs)):
            future = executor.submit(run_episode, env, e)
            result = future.result()
            rewards.append(result[0])
            epsilons.append(result[1])

            if (e + 1) % 1000 == 0:
                save_model(agent, temp=True)
                plot_training(rewards, epsilons, e + 1)

    final_model_save_path = os.path.join(current_dir, 'models', 'saved_model.pth')
    agent.save_model(final_model_save_path)
    print(f"Final model saved successfully at {final_model_save_path}.")

def plot_training(rewards, epsilons, episode):
    episodes = list(range(1, len(rewards) + 1))

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(episodes, rewards, label='Rewards per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Rewards over Episodes')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(episodes, [100 - (epsilon * 100) for epsilon in epsilons], label='Epsilon per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Epsilon (%)')
    plt.title('Epsilon over Episodes')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_plot.png')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, help="The model file to load for continued training")
    args = parser.parse_args()
    main(args)
