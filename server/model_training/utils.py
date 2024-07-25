import os
import json
import torch

hyperparameters_path = 'results/best_params.json'
model_save_path = 'results/model.pth'

def save_hyperparameters(params):
    if not os.path.exists(os.path.dirname(hyperparameters_path)):
        os.makedirs(os.path.dirname(hyperparameters_path))
    with open(hyperparameters_path, 'w') as f:
        json.dump(params, f)
    print(f'Hyperparameters saved to {hyperparameters_path}')

def load_hyperparameters():
    if os.path.exists(hyperparameters_path):
        with open(hyperparameters_path, 'r') as f:
            params = json.load(f)
        print(f'Loaded hyperparameters from {hyperparameters_path}')
        return params
    else:
        print('No hyperparameters file found, starting with default parameters.')
        return None

def save_model(agent, model_save_path):
    if not os.path.exists(os.path.dirname(model_save_path)):
        os.makedirs(os.path.dirname(model_save_path))
    torch.save(agent.policy_net.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

def load_model(agent, model_save_path):
    if os.path.exists(model_save_path):
        agent.policy_net.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
        agent.policy_net.eval()  # Set the model to evaluation mode
        print(f'Model loaded from {model_save_path}')
    else:
        print(f'No model found at {model_save_path}')
