from __future__ import annotations

import time

from python.environment import GoGridWorld


class PythonGridWorld:
    def __init__(
        self,
        rows: int = 6,
        cols: int = 6,
        step_reward: float = -1.0,
        obstacle_reward: float = -10.0,
        goal_reward: float = 100.0,
    ) -> None:
        self.rows = rows
        self.cols = cols

        self.step_reward = step_reward
        self.obstacle_reward = obstacle_reward
        self.goal_reward = goal_reward

        self.start = 0
        self.goal = rows * cols - 1

        self.agent = self.start

        self.obstacles = {
            (1, 1),
            (1, 3),
            (2, 3),
            (3, 1),
            (4, 1),
            (4, 4),
        }

    def reset(self) -> int:
        self.agent = self.start
        return self.agent

    def step(self, action: int) -> tuple[int, float, bool]:
        row = self.agent // self.cols
        col = self.agent % self.cols

        row_delta, col_delta = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1),
        }[action]

        new_row = row + row_delta
        new_col = col + col_delta

        if (
            new_row < 0
            or new_row >= self.rows
            or new_col < 0
            or new_col >= self.cols
        ):
            return self.agent, self.obstacle_reward, False

        if (new_row, new_col) in self.obstacles:
            return self.agent, self.obstacle_reward, False

        self.agent = new_row * self.cols + new_col

        if self.agent == self.goal:
            return self.agent, self.goal_reward, True

        return self.agent, self.step_reward, False


def benchmark_environment(
    environment,
    steps: int,
) -> float:
    actions = [0, 1, 2, 3]

    environment.reset()

    start_time = time.perf_counter()

    for index in range(steps):
        action = actions[index % len(actions)]

        _, _, done = environment.step(action)

        if done:
            environment.reset()

    elapsed = time.perf_counter() - start_time

    return steps / elapsed


def benchmark_go_batch(
    environment: GoGridWorld,
    steps: int,
    batch_size: int = 1000,
) -> float:
    actions = [0, 1, 2, 3]

    batches = steps // batch_size

    batch = [
        actions[index % len(actions)]
        for index in range(batch_size)
    ]

    start_time = time.perf_counter()

    for _ in range(batches):
        environment.reset()
        environment.batch_step(batch)

    elapsed = time.perf_counter() - start_time

    return steps / elapsed


def main() -> None:
    steps = 1_000_000

    python_environment = PythonGridWorld()

    go_environment = GoGridWorld(
        rows=6,
        cols=6,
        step_reward=-1.0,
        obstacle_reward=-10.0,
        goal_reward=100.0,
    )

    python_speed = benchmark_environment(
        python_environment,
        steps,
    )

    go_speed = benchmark_environment(
        go_environment,
        steps,
    )

    go_batch_speed = benchmark_go_batch(
        go_environment,
        steps,
    )

    go_speedup = go_speed / python_speed
    batch_speedup = go_batch_speed / python_speed

    print()
    print("Environment Benchmark")
    print("=" * 55)
    print(f"Total steps: {steps:,}")
    print()
    print(
        f"Python Environment:   "
        f"{python_speed:,.2f} steps/sec"
    )
    print(
        f"Go Environment:       "
        f"{go_speed:,.2f} steps/sec"
    )
    print(
        f"Go Batch Environment: "
        f"{go_batch_speed:,.2f} steps/sec"
    )
    print()
    print(
        f"Go speedup:       "
        f"{go_speedup:.2f}x"
    )
    print(
        f"Go Batch speedup: "
        f"{batch_speedup:.2f}x"
    )


if __name__ == "__main__":
    main()