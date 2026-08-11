from __future__ import annotations

import random

import numpy as np


class QLearningAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int = 4,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_table = np.zeros(
            (state_size, action_size),
            dtype=np.float64,
        )

    def choose_action(self, state: int) -> int:
        """Choose an action using epsilon-greedy exploration."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        return int(np.argmax(self.q_table[state]))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> None:
        """Update the Q-value using the Q-Learning update rule."""
        current_q = self.q_table[state, action]

        if done:
            target = reward
        else:
            best_next_q = np.max(self.q_table[next_state])
            target = reward + self.discount_factor * best_next_q

        self.q_table[state, action] += (
            self.learning_rate * (target - current_q)
        )

    def decay_epsilon(self) -> None:
        """Reduce exploration after each episode."""
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )

    def get_policy(self) -> np.ndarray:
        """Return the best action for every state."""
        return np.argmax(self.q_table, axis=1)

    def save(self, path: str) -> None:
        """Save the Q-table to disk."""
        np.save(path, self.q_table)

    def load(self, path: str) -> None:
        """Load a previously saved Q-table."""
        self.q_table = np.load(path)