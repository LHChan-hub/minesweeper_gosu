import matplotlib.pyplot as plt
import torch
import os

rewards = []

def log_episode(episode, reward):
    rewards.append(reward)
    if not os.path.exists('results'):
        os.makedirs('results')
    with open('results/training_log.txt', 'a') as f:
        f.write(f"Episode {episode}: Reward {reward}\n")

def plot_rewards(save_path='results/rewards_graph.png', num_points=20):
    rewards_tensor = torch.tensor(rewards, dtype=torch.float)
    total_episodes = len(rewards_tensor)
    interval = max(total_episodes // num_points, 1)
    
    averaged_rewards = []
    for i in range(0, total_episodes, interval):
        averaged_rewards.append(rewards_tensor[i:i + interval].mean().item())
    
    x_ticks = range(0, total_episodes + 1, interval)
    x_labels = range(0, total_episodes + 1, interval)

    # Correct length of x_labels
    x_labels = list(x_labels)
    if len(x_labels) > len(averaged_rewards):
        x_labels = x_labels[:len(averaged_rewards)]
    else:
        x_labels.extend([x_labels[-1]] * (len(averaged_rewards) - len(x_labels)))

    plt.plot(averaged_rewards)
    plt.title("Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.xticks(ticks=range(0, len(averaged_rewards)), labels=x_labels, rotation=45)
    plt.savefig(save_path)
    plt.close()
    
def log_test_results(test_log, path='results/test_log.txt'):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    
    with open(path, 'w') as f:
        for entry in test_log:
            f.write(f"[step : {entry['step']} , select : ({entry['x']}, {entry['y']}) , mine : {entry['mine']}]\n")
    print(f"Test results saved to {path}")

if __name__ == "__main__":
    if not os.path.exists('results'):
        os.makedirs('results')
    with open('results/training_log.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("Episode"):
                reward = float(line.split(": Reward ")[1])
                rewards.append(reward)
    plot_rewards()
