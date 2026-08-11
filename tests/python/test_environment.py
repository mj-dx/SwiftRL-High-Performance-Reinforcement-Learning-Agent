from python.environment import GoGridWorld


def test_environment_reset() -> None:
    environment = GoGridWorld()

    state = environment.reset()

    assert state == 0


def test_environment_step() -> None:
    environment = GoGridWorld()

    environment.reset()

    state, reward, done = environment.step(
        GoGridWorld.ACTION_RIGHT
    )

    assert state == 1
    assert reward == -1.0
    assert done is False


def test_batch_step() -> None:
    environment = GoGridWorld()

    environment.reset()

    actions = [
        GoGridWorld.ACTION_RIGHT,
        GoGridWorld.ACTION_RIGHT,
        GoGridWorld.ACTION_DOWN,
    ]

    states, rewards, dones = environment.batch_step(
        actions
    )

    assert len(states) == 3
    assert len(rewards) == 3
    assert len(dones) == 3

    assert states[0] == 1
    assert states[1] == 2
    assert states[2] == 8