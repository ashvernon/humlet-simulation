import pytest

np = pytest.importorskip("numpy")

from humlet_simulation.environment import Environment
from humlet_simulation.humlet import Humlet


def test_brain_forward_records_last_pass():
    env = Environment(100, 100)
    humlet = Humlet(env, seed=7)

    inputs = np.linspace(-1.0, 1.0, humlet.N_INPUTS)
    outputs = humlet._brain_forward(inputs)

    assert outputs.shape == (humlet.N_OUTPUTS,)
    assert np.allclose(humlet.last_inputs, inputs)
    assert humlet.last_hidden.shape == (humlet.N_HIDDEN,)
    assert humlet.last_outputs.shape == (humlet.N_OUTPUTS,)


def test_brain_learning_updates_weights_from_reward():
    env = Environment(120, 120)
    humlet = Humlet(env, seed=11)

    inputs = np.linspace(-0.5, 0.5, humlet.N_INPUTS)
    humlet._brain_forward(inputs)

    pre_W1 = humlet.brain["W1"].copy()
    pre_W2 = humlet.brain["W2"].copy()

    pre_state = {
        "energy": 50.0,
        "health": 55.0,
        "hunger": 0.6,
        "food_dist": 12.0,
    }
    post_state = {
        "energy": 60.0,  # gained energy
        "health": 56.0,
        "hunger": 0.3,  # reduced hunger
        "food_dist": 6.0,  # moved closer to food
        "collided": False,
    }

    humlet._update_brain_reward(pre_state, post_state, eat_signal=1.0, repro_signal=0.0)

    assert not np.allclose(humlet.brain["W1"], pre_W1)
    assert not np.allclose(humlet.brain["W2"], pre_W2)
    assert humlet.last_brain_reward > 0
    assert humlet.brain_fitness != 0
