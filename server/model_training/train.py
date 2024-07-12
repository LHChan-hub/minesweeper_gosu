import torch
import os
from gym_minesweeper.minesweeper_env import MinesweeperEnv
from dqn_agent import DQNAgent
from logger import Logger

def train():
    env = MinesweeperEnv()
    agent = DQNAgent(env, initial_tau=1.0, min_tau=0.01, decay_steps=40000, power=0.00005)  # Boltzmann 탐색 파라미터 설정
    logger = Logger(result_dir='results')

    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    num_episodes = 40000  # 총 에피소드 수 변경
    save_interval = 1000
    log_file = open("training_log.txt", "w")  # 로그 파일 열기

    for episode in range(num_episodes):
        state, _ = env.reset()
        agent.visited.clear()  # 에피소드 시작 시 방문 기록 초기화
        episode_reward = 0
        episode_loss = 0
        t = 0
        while True:
            action = agent.select_action(state)
            if action is None:
                state, _ = env.reset()  # 이미 방문한 상태를 선택한 경우 초기화
                continue

            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            loss = agent.update(episode)
            if loss:
                episode_loss += loss
                t += 1

            if done:
                if episode_reward == 0:
                    state, _ = env.reset()
                    continue
                avg_loss = episode_loss / t if t > 0 else 0
                logger.log_episode(episode_reward, agent.tau, avg_loss)  # tau 값 로깅

                # 모든 에피소드마다 로그 출력
                print(f"Episode {episode} ended with reward {episode_reward:.2f}, tau {agent.tau:.5f}, loss {avg_loss:.2f}")
                # 로그 파일에 기록
                log_file.write(f"Episode {episode} ended with reward {episode_reward:.2f}, tau {agent.tau:.5f}, loss {avg_loss:.2f}\n")

                break

        if (episode + 1) % save_interval == 0:
            logger.plot_and_save(f'results_{episode + 1}')
            model_path = os.path.join(model_dir, f'dqn_agent_{episode + 1}.pth')
            torch.save(agent.q_network.state_dict(), model_path)
            print(f"Saved interim results and model at episode {episode + 1}")

    logger.plot_and_save('final_results')
    model_path = os.path.join(model_dir, 'dqn_agent_final.pth')
    torch.save(agent.q_network.state_dict(), model_path)

    log_file.close()  # 로그 파일 닫기

if __name__ == "__main__":
    train()
