import numpy as np

from python.agents.q_learning import QLearningAgent


def test_agent_initialization() -> None:
    agent = QLearningAgent(
        state_size=36,
        action_size=4,
    )

    assert agent.q_table.shape == (36, 4)
    assert np.all(agent.q_table == 0.0)


def test_action_is_valid() -> None:
    agent = QLearningAgent(
        state_size=36,
        action_size=4,
    )

    action = agent.choose_action(0)

    assert 0 <= action < 4


def test_q_value_update() -> None:
    agent = QLearningAgent(
        state_size=36,
        action_size=4,
        learning_rate=0.1,
        discount_factor=0.99,
    )

    agent.update(
        state=0,
        action=3,
        reward=10.0,
        next_state=1,
        done=True,
    )

    assert agent.q_table[0, 3] > 0.0


def test_epsilon_decay() -> None:
    agent = QLearningAgent(
        state_size=36,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.5,
    )

    agent.decay_epsilon()

    assert agent.epsilon == 0.5