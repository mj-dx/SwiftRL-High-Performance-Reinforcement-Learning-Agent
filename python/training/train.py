from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt

from python.agents.q_learning import QLearningAgent
from python.environment import GoGridWorld


def train(
    episodes: int = 500,
    max_steps: int = 100,
) -> QLearningAgent:
    environment = GoGridWorld(
        rows=6,
        cols=6,
        step_reward=-1.0,
        obstacle_reward=-10.0,
        goal_reward=100.0,
    )

    agent = QLearningAgent(
        state_size=environment.rows * environment.cols,
        action_size=4,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
    )

    rewards_per_episode = []
    steps_per_episode = []

    start_time = time.perf_counter()

    for episode in range(episodes):
        state = environment.reset()

        total_reward = 0.0
        steps = 0

        for _ in range(max_steps):
            action = agent.choose_action(state)

            next_state, reward, done = environment.step(action)

            agent.update(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            )

            state = next_state
            total_reward += reward
            steps += 1

            if done:
                break

        agent.decay_epsilon()

        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)

        if (episode + 1) % 50 == 0:
            print(
                f"Episode: {episode + 1}/{episodes} | "
                f"Reward: {total_reward:.2f} | "
                f"Steps: {steps} | "
                f"Epsilon: {agent.epsilon:.4f}"
            )

    elapsed_time = time.perf_counter() - start_time

    print()
    print(f"Training completed in {elapsed_time:.4f} seconds")
    print(f"Episodes per second: {episodes / elapsed_time:.2f}")

    save_training_plot(rewards_per_episode)
    model_dir = Path("results/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "q_learning.npy"
    agent.save(str(model_path))

    print(f"Model saved to: {model_path}")

    return agent


def save_training_plot(rewards: list[float]) -> None:
    output_dir = Path("results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Q-Learning Training Reward")
    plt.grid(True)
    plt.tight_layout()

    output_path = output_dir / "q_learning_training.png"
    plt.savefig(output_path)
    plt.close()

    print(f"Training plot saved to: {output_path}")


if __name__ == "__main__":
    train()