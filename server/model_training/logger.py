import matplotlib.pyplot as plt
import numpy as np
import os

class Logger:
    def __init__(self, result_dir='results'):
        self.result_dir = result_dir
        self.rewards = []
        self.taus = []
        self.losses = []

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

    def log_episode(self, reward, tau, loss):
        self.rewards.append(reward)
        self.taus.append(tau)
        self.losses.append(loss)

    def plot_and_save(self, filename):
        fig, axs = plt.subplots(3, 1, figsize=(12, 18))

        # Rewards per Episode
        axs[0].plot(self.rewards)
        axs[0].set_title('Rewards per Episode')
        axs[0].set_xlabel('Episode')
        axs[0].set_ylabel('Total Reward')

        # Tau per Episode
        axs[1].plot(self.taus)
        axs[1].set_title('Tau per Episode')
        axs[1].set_xlabel('Episode')
        axs[1].set_ylabel('Tau')

        # Loss per Episode
        axs[2].plot(self.losses)
        axs[2].set_title('Loss per Episode')
        axs[2].set_xlabel('Episode')
        axs[2].set_ylabel('Loss')

        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, f"{filename}.png"))
        plt.close(fig)
