from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from python.agents.q_learning import QLearningAgent
from python.environment import GoGridWorld


ACTION_ARROWS = {
    0: "↑",
    1: "↓",
    2: "←",
    3: "→",
}


def decode_state(state: int, cols: int) -> tuple[int, int]:
    return state // cols, state % cols


def visualize() -> None:
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
            f"Model not found: {model_path}. Run training first."
        )

    agent.load(str(model_path))
    agent.epsilon = 0.0

    state = environment.reset()

    path = [state]
    actions = []

    goal_reached = False

    for _ in range(100):
        action = agent.choose_action(state)

        next_state, reward, done = environment.step(action)

        actions.append(action)
        path.append(next_state)

        state = next_state

        if done:
            goal_reached = reward > 0
            break

    figure, axis = plt.subplots(figsize=(8, 8))

    grid = np.zeros(
        (environment.rows, environment.cols),
        dtype=float,
    )

    axis.imshow(
        grid,
        cmap="gray_r",
        vmin=-1,
        vmax=1,
    )

    start_row, start_col = decode_state(
        path[0],
        environment.cols,
    )

    final_row, final_col = decode_state(
        path[-1],
        environment.cols,
    )

    axis.text(
        start_col,
        start_row,
        "S",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
    )

    axis.text(
        final_col,
        final_row,
        "G" if goal_reached else "X",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
    )

    for index, current_state in enumerate(path[:-1]):
        row, col = decode_state(
            current_state,
            environment.cols,
        )

        action = actions[index]

        axis.text(
            col,
            row,
            ACTION_ARROWS[action],
            ha="center",
            va="center",
            fontsize=24,
        )

    axis.set_xticks(
        np.arange(-0.5, environment.cols, 1),
        minor=True,
    )

    axis.set_yticks(
        np.arange(-0.5, environment.rows, 1),
        minor=True,
    )

    axis.grid(
        which="minor",
        linewidth=1,
    )

    axis.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    axis.set_xticks(
        np.arange(environment.cols)
    )

    axis.set_yticks(
        np.arange(environment.rows)
    )

    axis.set_title(
        "Learned Q-Learning Policy"
    )

    output_dir = Path("results/plots")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir / "learned_policy.png"
    )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

    print()
    print("Evaluation Visualization")
    print("=" * 40)
    print(f"Path length: {len(path) - 1}")
    print(f"Goal reached: {goal_reached}")
    print(f"Final state: {path[-1]}")
    print(f"Visualization saved to: {output_path}")


if __name__ == "__main__":
    visualize()