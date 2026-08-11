from __future__ import annotations

from pathlib import Path

from python.agents.q_learning import QLearningAgent
from python.environment import GoGridWorld


def decode_state(state: int, cols: int) -> tuple[int, int]:
    return state // cols, state % cols


def evaluate(
    episodes: int = 100,
    max_steps: int = 100,
) -> None:
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
    )

    model_path = Path("results/models/q_learning.npy")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. "
            "Run training first."
        )

    agent.load(str(model_path))

    agent.epsilon = 0.0

    successful_episodes = 0
    total_rewards = []
    total_steps = []

    best_path = None
    best_steps = float("inf")

    for episode in range(episodes):
        state = environment.reset()

        episode_reward = 0.0
        path = [decode_state(state, environment.cols)]

        for _ in range(max_steps):
            action = agent.choose_action(state)

            next_state, reward, done = environment.step(action)

            state = next_state
            episode_reward += reward

            path.append(
                decode_state(state, environment.cols)
            )

            if done:
                successful_episodes += 1

                if len(path) - 1 < best_steps:
                    best_steps = len(path) - 1
                    best_path = path.copy()

                break

        total_rewards.append(episode_reward)
        total_steps.append(len(path) - 1)

    success_rate = (
        successful_episodes / episodes
    ) * 100

    average_reward = (
        sum(total_rewards) / len(total_rewards)
    )

    average_steps = (
        sum(total_steps) / len(total_steps)
    )

    print()
    print("Evaluation Results")
    print("=" * 40)
    print(f"Episodes: {episodes}")
    print(f"Successful Episodes: {successful_episodes}")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Average Reward: {average_reward:.2f}")
    print(f"Average Steps: {average_steps:.2f}")

    if best_path is not None:
        print(f"Best Steps: {int(best_steps)}")
        print()
        print("Best Path:")

        for index, position in enumerate(best_path):
            print(f"{index}: {position}")


if __name__ == "__main__":
    evaluate()