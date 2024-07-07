import sys
import os
import argparse
import glob

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

from dqn_agent import DQNAgent, device

def main(args):
    env = gym.make('Minesweeper-v0')
    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_size = env.action_space.n
    agent = DQNAgent(state_dim=state_size, action_dim=action_size)

    if args.model_file:
        model_path = os.path.join(current_dir, 'models', args.model_file)
        if os.path.exists(model_path):
            agent.load(model_path)
            print(f"Model {args.model_file} loaded successfully.")
        else:
            print(f"No model found at {model_path}. Starting training from scratch.")
    else:
        print("No model file specified. Starting training from scratch.")

    def save_model(agent, episode):
        model_save_path = os.path.join(current_dir, 'models', f'saved_model_{episode}.pth')
        agent.save(model_save_path)
        print(f"Model saved successfully at {model_save_path}.")
        model_files = sorted(glob.glob(os.path.join(current_dir, 'models', 'saved_model_*.pth')))
        if len(model_files) > 50:
            os.remove(model_files[0])
            print(f"Removed old model: {model_files[0]}")

    for e in range(100000):
        state = env.reset().flatten().to(device)
        selected_positions = set()
        for time in range(500):
            action = agent.act(state, selected_positions)
            x, y = divmod(action, int(env.width))
            selected_positions.add((x, y))
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.flatten().to(device)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"episode: {e + 1}/100000, score: {reward}, epsilon: {agent.epsilon:.3f}")
                break
            if len(agent.memory) > agent.batch_size:
                agent.replay()

        if (e + 1) % 500 == 0:
            save_model(agent, e + 1)

    final_model_save_path = os.path.join(current_dir, 'models', 'final_saved_model.pth')
    agent.save(final_model_save_path)
    print(f"Final model saved successfully at {final_model_save_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, help="The model file to load for continued training")
    args = parser.parse_args()
    main(args)
