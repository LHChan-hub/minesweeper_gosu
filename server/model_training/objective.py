import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from env import MineSweeperEnv
from agent import DQNAgent
from train import train
from utils import save_hyperparameters

def objective(trial):
    gamma = trial.suggest_float("gamma", 0.8, 0.999)
    epsilon = trial.suggest_float("epsilon", 0.01, 1.0)
    epsilon_decay = trial.suggest_float("epsilon_decay", 0.99, 0.99999)
    epsilon_min = trial.suggest_float("epsilon_min", 0.001, 0.1)
    lr = trial.suggest_float("lr", 1e-6, 1e-2)
    batch_size = trial.suggest_int("batch_size", 16, 512)
    epsilon_reset_threshold = trial.suggest_int("epsilon_reset_threshold", 10, 500)
    epsilon_reset_value = trial.suggest_float("epsilon_reset_value", 0.01, 1.0)

    fc1_units = trial.suggest_int("fc1_units", 32, 1024)
    fc2_units = trial.suggest_int("fc2_units", 32, 1024)
    fc3_units = trial.suggest_int("fc3_units", 32, 1024)

    penalty_mine = trial.suggest_int("penalty_mine", -100, -10)
    reward_empty = trial.suggest_int("reward_empty", 1, 20)
    penalty_revisit = trial.suggest_int("penalty_revisit", -20, -1)
    proximity_reward = trial.suggest_int("proximity_reward", 1, 10)
    initial_exploration_bonus = trial.suggest_int("initial_exploration_bonus", 1, 10)
    periodic_bonus = trial.suggest_int("periodic_bonus", 1, 20)
    win_reward = trial.suggest_int("win_reward", 10, 200)
    step_penalty = trial.suggest_int("step_penalty", -50, -1)

    env = MineSweeperEnv(width=10, height=10, n_mines=10,
                         penalty_mine=penalty_mine, reward_empty=reward_empty, penalty_revisit=penalty_revisit,
                         proximity_reward=proximity_reward, initial_exploration_bonus=initial_exploration_bonus,
                         periodic_bonus=periodic_bonus, win_reward=win_reward, step_penalty=step_penalty)
    state_dim = env.observation_space[0]
    action_dim = env.action_space

    agent = DQNAgent(state_dim, action_dim, gamma, epsilon, epsilon_decay, epsilon_min, lr, epsilon_reset_threshold, epsilon_reset_value, fc1_units, fc2_units, fc3_units)
    
    max_episodes = 1000
    average_reward = train(agent, env, batch_size=batch_size, save_interval=100, update_interval=100, resume=False)
    return average_reward

if __name__ == "__main__":
    storage_path = 'results/optuna_study.db'
    if not os.path.exists(storage_path):
        study = optuna.create_study(direction="maximize",
                                    sampler=TPESampler(),
                                    pruner=HyperbandPruner(),
                                    storage=f"sqlite:///{storage_path}",
                                    study_name="dqn_minesweeper")
        study.optimize(objective, n_trials=200)

        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        save_hyperparameters(trial.params)
    else:
        study = optuna.load_study(study_name="dqn_minesweeper", storage=f"sqlite:///{storage_path}")
