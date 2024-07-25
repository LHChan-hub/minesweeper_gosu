import torch
import os

checkpoint_path = 'results/checkpoints/checkpoint.pth'

def save_checkpoint(agent, episode, total_reward, total_steps):
    if not os.path.exists(os.path.dirname(checkpoint_path)):
        os.makedirs(os.path.dirname(checkpoint_path))
    torch.save({
        'episode': episode,
        'total_steps': total_steps,
        'best_total_reward': total_reward,
        'policy_net_state_dict': agent.policy_net.state_dict(),
        'target_net_state_dict': agent.target_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
    }, checkpoint_path)
    print(f'Checkpoint saved at episode {episode} with total reward {total_reward}')

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
