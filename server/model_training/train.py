import sys
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import json
import itertools

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
from hp_combination import HPCombination, hp_comb

results_dir = os.path.join(current_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

final_scores = []  # 최종 스코어를 저장할 리스트

def save_board_and_selections(board, selections, episode, filename):
    with open(filename, 'a') as f:
        f.write(f"Episode {episode}:\n")
        f.write("Final Board:\n")
        for row in board:
            f.write(' '.join(str(cell) for cell in row) + '\n')
        f.write("\nModel Selections:\n")
        for selection in selections:
            x, y = selection if len(selection) == 2 else (selection[0], selection[1])
            is_mine = board[x][y] == -1
            f.write(f"({x}, {y}, {'Select'}): {'Mine' if is_mine else 'Safe'}\n")
        f.write("\n\n")

def plot_hyperparameter_results():
    hp_ids = [f'hp{str(i).zfill(3)}' for i in range(len(final_scores))]
    plt.figure(figsize=(10, 5))
    plt.plot(hp_ids, final_scores, marker='o')
    plt.xlabel('Hyperparameter ID')
    plt.ylabel('Final Average Score')
    plt.title('Final Average Score vs Hyperparameter ID')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'final_scores.png'))
    plt.close()

def main(args):
    env = gym.make('Minesweeper-v0')
    hp_idx = 0

    while hp_idx < 200:  # 최대 200개의 하이퍼파라미터 조합 테스트
        params = hp_comb.get_current_params()
        if params is None:
            hp_comb.setup_next_test()
            params = hp_comb.get_current_params()
        
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        agent = BootstrappedNoisyDQNAgent(state_dim=state_size, action_dim=action_size, **params)

        if args.model_file:
            model_path = os.path.join(current_dir, 'saved_model.pth')
            if os.path.exists(model_path):
                agent.load_model(model_path)
                print(f"Model {args.model_file} loaded successfully.")
            else:
                print(f"No model found at {model_path}. Starting training from scratch.")
        else:
            print("No model file specified. Starting training from scratch.")

        def save_model(agent):
            model_save_path = os.path.join(results_dir, f'saved_model_hp{hp_idx:03d}.pth')
            agent.save_model(model_save_path)
            print(f"Model saved successfully at {model_save_path}.")

        rewards = []
        epsilons = []

        def run_episode(agent, env, e):
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32).to(device)
            selected_positions = set()
            total_reward = 0
            terminated = False
            while not terminated:
                action = agent.act(state, selected_positions)
                x, y = env.decode_action(action)

                if x >= env.width or y >= env.height:
                    print(f"Invalid action: {action} (x: {x}, y: {y}), skipping")
                    continue

                selected_positions.add((x, y))

                next_state, reward, terminated, _, _ = env.step(action)
                next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
                if reward >= 10:  # 보상이 10 이상인 경우 가중치를 추가로 부여
                    reward *= 2
                error = abs(reward + agent.gamma * torch.max(agent.models[0](next_state)) - torch.max(agent.models[0](state))).item()
                agent.remember(state, action, reward, next_state, terminated, error)
                state = next_state
                total_reward += reward
                if len(agent.memory.buffer) > agent.batch_size:
                    agent.replay()
            return total_reward, agent.epsilon, env.board, selected_positions

        log_filename = os.path.join(results_dir, f'episode_logs_hp{hp_idx:03d}.txt')
        if os.path.exists(log_filename):
            os.remove(log_filename)

        for e in range(hp_comb.max_episodes):
            total_reward, epsilon, board, selections = run_episode(agent, env, e)
            rewards.append(total_reward)
            epsilons.append(epsilon)

            if e < 50:
                save_board_and_selections(board, selections, e + 1, log_filename)

            print(f"Episode: {e + 1}, Total Reward: {total_reward}, Epsilon: {epsilon}")

        recent_rewards = rewards[-50:]
        average_score = sum(recent_rewards) / 50
        final_scores.append(average_score)  # 최종 스코어 저장

        hp_comb.save_results(hp_idx, rewards)

        plot_training(rewards, epsilons, hp_comb.max_episodes, hp_idx)
        plot_hyperparameter_results()  # 최종 스코어 그래프 업데이트

        if average_score < 50:
            print(f"Low average score detected. Restarting training with new parameters.")
            hp_comb.update_params(recent_rewards)
            hp_comb.reset()
        else:
            save_model(agent)
            print(f"Final model saved successfully at {results_dir}/saved_model_hp{hp_idx:03d}.pth.")
            break

        hp_idx += 1

def plot_training(rewards, epsilons, episode, hp_idx):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(rewards, label='Rewards per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Rewards over Episodes')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epsilons, label='Epsilon per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Epsilon')
    plt.title('Epsilon over Episodes')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'training_plot_hp{hp_idx:03d}.png'))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, default=None, help='Path to the model file to load')
    args = parser.parse_args()
    main(args)
