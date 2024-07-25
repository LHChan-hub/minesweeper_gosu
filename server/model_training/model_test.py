import torch
from env import MineSweeperEnv
from agent import DQNAgent
from log import log_test_results
import os
import json

checkpoint_path = 'results/checkpoints/checkpoint.pth'
hyperparameters_path = 'results/best_params.json'

def load_checkpoint(agent):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f'Loaded checkpoint from episode {checkpoint["episode"]}')
        return checkpoint['episode'], checkpoint['total_steps'], checkpoint['best_total_reward']
    else:
        print('No checkpoint found, starting from scratch.')
        return 0, 0, float('-inf')

def load_hyperparameters():
    if os.path.exists(hyperparameters_path):
        with open(hyperparameters_path, 'r') as f:
            params = json.load(f)
        print(f'Loaded hyperparameters from {hyperparameters_path}')
        return params
    else:
        print('No hyperparameters file found, starting with default parameters.')
        return None

def evaluate_model(agent, env, num_episodes=100):
    total_steps_list = []

    for episode in range(num_episodes):
        state = env.reset().cuda()
        steps = 0
        visited_cells = set()
        test_log = []

        while True:
            action = agent.select_action(state)
            x, y = divmod(action, env.width)

            if (x, y) in visited_cells:
                continue
            visited_cells.add((x, y))

            next_state, reward, done, _ = env.step(action)
            next_state = next_state.cuda()
            state = next_state
            steps += 1

            test_log.append({
                "step": steps,
                "x": x,
                "y": y,
                "mine": reward == env.penalty_mine
            })

            if done:
                break

        total_steps_list.append(steps)

        if episode == num_episodes - 1:  # Save the last episode's log
            log_test_results(test_log)

    average_steps = sum(total_steps_list) / len(total_steps_list)
    print(f"Evaluation over {num_episodes} episodes")
    print(f"Average Steps: {average_steps}")

    return average_steps

if __name__ == "__main__":
    best_params = load_hyperparameters()
    
    if best_params is None:
        best_params = {
            "penalty_mine": -50,
            "reward_empty": 10,
            "penalty_revisit": -5,
            "proximity_reward": 2,
            "initial_exploration_bonus": 8,
            "periodic_bonus": 15,
            "win_reward": 113,
            "step_penalty": -40,
            "gamma": 0.99,
            "epsilon": 1.0,
            "epsilon_decay": 0.99,
            "epsilon_min": 0.1,
            "lr": 1e-3,
            "batch_size": 64,
            "epsilon_reset_threshold": 100,
            "epsilon_reset_value": 1.0,
            "fc1_units": 200,
            "fc2_units": 433,
            "fc3_units": 977
        }

    env = MineSweeperEnv(width=10, height=10, n_mines=10,
                         penalty_mine=best_params['penalty_mine'], reward_empty=best_params['reward_empty'], penalty_revisit=best_params['penalty_revisit'],
                         proximity_reward=best_params['proximity_reward'], initial_exploration_bonus=best_params['initial_exploration_bonus'],
                         periodic_bonus=best_params['periodic_bonus'], win_reward=best_params['win_reward'], step_penalty=best_params['step_penalty'])
    state_dim = env.observation_space[0]
    action_dim = env.action_space

    agent = DQNAgent(state_dim, action_dim, gamma=best_params['gamma'], epsilon=best_params['epsilon'], epsilon_decay=best_params['epsilon_decay'], epsilon_min=best_params['epsilon_min'], lr=best_params['lr'], epsilon_reset_threshold=best_params['epsilon_reset_threshold'], epsilon_reset_value=best_params['epsilon_reset_value'], fc1_units=best_params['fc1_units'], fc2_units=best_params['fc2_units'], fc3_units=best_params['fc3_units'])
    
    load_checkpoint(agent)
    evaluate_model(agent, env)
