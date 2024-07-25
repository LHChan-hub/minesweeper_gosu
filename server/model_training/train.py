import torch
from env import MineSweeperEnv
from agent import DQNAgent
import log
from checkpoint import save_checkpoint, load_checkpoint
from utils import save_model, load_hyperparameters
from model_test import evaluate_model

def train(agent, env, batch_size, save_interval, update_interval, resume=False):
    episode = 0
    total_steps = 0
    best_total_reward = float('-inf')

    if resume:
        episode, total_steps, best_total_reward = load_checkpoint(agent)

    try:
        while True:
            state = env.reset()
            total_reward = 0

            while True:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward

                agent.store_transition(state, action, next_state if not done else None, reward)
                state = next_state

                agent.optimize_model(batch_size)

                if done:
                    agent.update_target()
                    break

                total_steps += 1

            average_reward = agent.update_rewards_window(total_reward)

            print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}, Average Reward: {average_reward}")

            if total_reward > best_total_reward:
                best_total_reward = total_reward
                save_checkpoint(agent, episode, total_reward, total_steps)

                # Evaluate model when achieving new best reward
                avg_eval_steps = evaluate_model(agent, env)
                print(f"Evaluation: Average Steps: {avg_eval_steps}")

                # Adjust hyperparameters based on evaluation results
                if avg_eval_steps > 70:  # Example condition
                    agent.epsilon = max(agent.epsilon * 0.95, agent.epsilon_min)  # Adjust epsilon
                    agent.lr *= 1.1  # Adjust learning rate
                    print("Adjusting hyperparameters: Increased learning rate and decreased epsilon")

            if (episode + 1) % update_interval == 0:
                log.plot_rewards()
                print(f"Graph updated at episode {episode + 1}")

            if (episode + 1) % save_interval == 0:
                save_checkpoint(agent, episode, total_reward, total_steps)

            episode += 1

    except KeyboardInterrupt:
        print("Training interrupted. Saving current state...")
        save_checkpoint(agent, episode, total_reward, total_steps)
        save_model(agent, 'results/model.pth')
        print("Checkpoint and model saved. Exiting...")

if __name__ == "__main__":
    best_params = load_hyperparameters()
    if best_params is None:
        print('No hyperparameters found, cannot proceed with training.')
    else:
        env = MineSweeperEnv(width=10, height=10, n_mines=10,
                             penalty_mine=best_params['penalty_mine'], reward_empty=best_params['reward_empty'], penalty_revisit=best_params['penalty_revisit'],
                             proximity_reward=best_params['proximity_reward'], initial_exploration_bonus=best_params['initial_exploration_bonus'],
                             periodic_bonus=best_params['periodic_bonus'], win_reward=best_params['win_reward'], step_penalty=best_params['step_penalty'],
                             variant="extended_steps")
        state_dim = env.observation_space[0]
        action_dim = env.action_space

        agent = DQNAgent(state_dim, action_dim, gamma=best_params['gamma'], epsilon=best_params['epsilon'], epsilon_decay=best_params['epsilon_decay'], epsilon_min=best_params['epsilon_min'], lr=best_params['lr'], epsilon_reset_threshold=best_params['epsilon_reset_threshold'], epsilon_reset_value=best_params['epsilon_reset_value'], fc1_units=best_params['fc1_units'], fc2_units=best_params['fc2_units'], fc3_units=best_params['fc3_units'])
        
        train(agent, env, batch_size=best_params['batch_size'], save_interval=1000, update_interval=1000, resume=True)
