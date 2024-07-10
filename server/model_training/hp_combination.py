import numpy as np
import json
import os

class HPCombination:
    def __init__(self, initial_params, step_sizes, param_ranges, max_episodes=500):
        self.params = initial_params
        self.step_sizes = step_sizes
        self.param_ranges = param_ranges
        self.max_episodes = max_episodes
        self.current_param = None
        self.current_index = 0
        self.results_dir = 'results'
        os.makedirs(self.results_dir, exist_ok=True)
        self.setup_next_test()

    def setup_next_test(self):
        if self.current_param is None:
            self.current_param = list(self.params.keys())[0]
        else:
            param_names = list(self.params.keys())
            current_param_index = param_names.index(self.current_param)
            if current_param_index < len(param_names) - 1:
                self.current_param = param_names[current_param_index + 1]
            else:
                self.current_param = param_names[0]
        
        self.current_index = 0
        self.test_combinations = [
            self.params[self.current_param] - self.step_sizes[self.current_param],
            self.params[self.current_param],
            self.params[self.current_param] + self.step_sizes[self.current_param]
        ]

    def update_params(self, scores):
        best_score_index = np.argmax(scores)
        if best_score_index >= len(self.test_combinations):
            print(f"Invalid best_score_index: {best_score_index}, resetting to 0")
            best_score_index = 0
        
        best_param_value = self.test_combinations[best_score_index]
        
        if best_score_index == 0:
            self.params[self.current_param] -= self.step_sizes[self.current_param]
            self.step_sizes[self.current_param] /= 2
        elif best_score_index == 2:
            self.params[self.current_param] += self.step_sizes[self.current_param]
            self.step_sizes[self.current_param] /= 2
        
        self.params[self.current_param] = max(
            self.param_ranges[self.current_param][0], 
            min(self.params[self.current_param], self.param_ranges[self.current_param][1])
        )

    def save_results(self, hp_idx, scores):
        results_path = os.path.join(self.results_dir, f'hyperparameters_hp{hp_idx:03d}.txt')
        with open(results_path, 'w') as f:
            f.write(json.dumps(self.params, indent=4))
            f.write(f"\nAverage Score: {np.mean(scores[-50:])}\n")
        
    def get_current_params(self):
        current_params = self.params.copy()
        current_params[self.current_param] = self.test_combinations[self.current_index]
        return current_params

    def next_combination(self):
        self.current_index += 1
        if self.current_index >= len(self.test_combinations):
            return None
        return self.get_current_params()
    
    def reset(self):
        self.current_index = 0

# Initial hyperparameters setup
initial_params = {
    'hidden_dim': 256,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.99995
}
step_sizes = {
    'hidden_dim': 64,
    'gamma': 0.01,
    'epsilon_start': 0.1,
    'epsilon_end': 0.005,
    'epsilon_decay': 0.00001
}
param_ranges = {
    'hidden_dim': (128, 512),
    'gamma': (0.9, 0.999),
    'epsilon_start': (0.5, 1.5),
    'epsilon_end': (0.001, 0.1),
    'epsilon_decay': (0.00001, 0.1)
}

hp_comb = HPCombination(initial_params, step_sizes, param_ranges)
