from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


Position = Tuple[int, int]


@dataclass
class GridWorldConfig:
    rows: int = 6
    cols: int = 6
    step_reward: float = -1.0
    obstacle_reward: float = -10.0
    goal_reward: float = 100.0


class GridWorld:
    ACTIONS = {
        0: (-1, 0),  # Up
        1: (1, 0),   # Down
        2: (0, -1),  # Left
        3: (0, 1),   # Right
    }

    def __init__(self, config: GridWorldConfig | None = None) -> None:
        self.config = config or GridWorldConfig()

        self.rows = self.config.rows
        self.cols = self.config.cols

        self.start: Position = (0, 0)
        self.goal: Position = (self.rows - 1, self.cols - 1)

        self.obstacles: set[Position] = {
            (1, 1),
            (1, 3),
            (2, 3),
            (3, 1),
            (4, 1),
            (4, 4),
        }

        self.agent_position: Position = self.start
        self.done = False

    def reset(self) -> Position:
        """Reset the environment and return the initial state."""
        self.agent_position = self.start
        self.done = False

        return self.get_state()

    def get_state(self) -> Position:
        """Return the current agent position."""
        return self.agent_position

    def step(self, action: int) -> tuple[Position, float, bool]:
        """
        Execute an action.

        Returns:
            next_state: The new agent position.
            reward: Reward received after the action.
            done: Whether the episode has finished.
        """
        if self.done:
            raise RuntimeError("Episode has already finished. Call reset().")

        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action: {action}")

        row, col = self.agent_position
        row_delta, col_delta = self.ACTIONS[action]

        new_position = (
            row + row_delta,
            col + col_delta,
        )

        if not self._is_valid_position(new_position):
            return self.agent_position, self.config.obstacle_reward, False

        self.agent_position = new_position

        if self.agent_position == self.goal:
            self.done = True
            return self.agent_position, self.config.goal_reward, True

        return self.agent_position, self.config.step_reward, False

    def _is_valid_position(self, position: Position) -> bool:
        """Check whether a position is inside the grid and not an obstacle."""
        row, col = position

        if row < 0 or row >= self.rows:
            return False

        if col < 0 or col >= self.cols:
            return False

        if position in self.obstacles:
            return False

        return True

    def render(self) -> None:
        """Print the current environment state."""
        grid = np.full(
            (self.rows, self.cols),
            ".",
            dtype="<U1",
        )

        for obstacle in self.obstacles:
            row, col = obstacle
            grid[row, col] = "#"

        goal_row, goal_col = self.goal
        grid[goal_row, goal_col] = "G"

        agent_row, agent_col = self.agent_position
        grid[agent_row, agent_col] = "A"

        print()
        for row in grid:
            print(" ".join(row))
        print()


if __name__ == "__main__":
    environment = GridWorld()

    environment.reset()
    environment.render()

    actions = [3, 3, 1, 1, 1, 1, 3, 3, 1]

    for action in actions:
        state, reward, done = environment.step(action)

        print(
            f"Action: {action} | "
            f"State: {state} | "
            f"Reward: {reward} | "
            f"Done: {done}"
        )

        environment.render()

        if done:
            break