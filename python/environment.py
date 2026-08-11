from __future__ import annotations

import ctypes
from pathlib import Path


class GoGridWorld:
    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3

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

        library_path = (
            Path(__file__).resolve().parent.parent
            / "go"
            / "bindings"
            / "gridworld.dll"
        )

        if not library_path.exists():
            raise FileNotFoundError(
                f"Go shared library not found: {library_path}"
            )

        self.library = ctypes.CDLL(str(library_path))

        self._configure_functions()

        self.library.CreateEnvironment(
            rows,
            cols,
            step_reward,
            obstacle_reward,
            goal_reward,
        )

    def _configure_functions(self) -> None:
        self.library.CreateEnvironment.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
        ]
        self.library.CreateEnvironment.restype = None

        self.library.ResetEnvironment.argtypes = []
        self.library.ResetEnvironment.restype = ctypes.c_int

        self.library.StepEnvironment.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
        ]
        self.library.StepEnvironment.restype = ctypes.c_int

        self.library.RunBatch.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
        ]

        self.library.RunBatch.restype = ctypes.c_int
        
    def reset(self) -> int:
        state = self.library.ResetEnvironment()

        if state < 0:
            raise RuntimeError("Failed to reset Go environment.")

        return state

    def step(self, action: int) -> tuple[int, float, bool]:
        if action not in range(4):
            raise ValueError(f"Invalid action: {action}")

        reward = ctypes.c_double()
        done = ctypes.c_int()

        state = self.library.StepEnvironment(
            action,
            ctypes.byref(reward),
            ctypes.byref(done),
        )

        if state < 0:
            raise RuntimeError("Failed to execute environment step.")

        return state, reward.value, bool(done.value)
    def batch_step(
        self,
        actions: list[int],
    ) -> tuple[list[int], list[float], list[bool]]:
        if not actions:
            return [], [], []

        if any(action not in range(4) for action in actions):
            raise ValueError("Invalid action in batch.")

        action_array = (ctypes.c_int * len(actions))(
            *actions
        )

        states_array = (
            ctypes.c_int * len(actions)
        )()

        rewards_array = (
            ctypes.c_double * len(actions)
        )()

        dones_array = (
            ctypes.c_int * len(actions)
        )()

        result = self.library.RunBatch(
            action_array,
            len(actions),
            states_array,
            rewards_array,
            dones_array,
        )

        if result < 0:
            raise RuntimeError(
                "Failed to execute batch."
            )

        states = list(states_array)
        rewards = list(rewards_array)
        dones = [
            bool(value)
            for value in dones_array
        ]

        return states, rewards, dones

if __name__ == "__main__":
    environment = GoGridWorld()

    state = environment.reset()

    print(f"Initial state: {state}")

    actions = [
        GoGridWorld.ACTION_RIGHT,
        GoGridWorld.ACTION_RIGHT,
        GoGridWorld.ACTION_DOWN,
        GoGridWorld.ACTION_DOWN,
    ]

    for action in actions:
        state, reward, done = environment.step(action)

        print(
            f"Action: {action} | "
            f"State: {state} | "
            f"Reward: {reward} | "
            f"Done: {done}"
        )

        if done:
            break