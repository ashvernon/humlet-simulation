import copy
import math

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


def test_rest_output_slows_agent_and_reduces_burn():
    env = Environment(80, 80)

    genome = {
        "metabolism_rate": 0.03,
        "speed_trait": 1.0,
        "sense_range": 60.0,
        "aggression": 0.1,
        "sociability": 0.5,
        "lifespan": 5000,
        "curiosity_trait": 0.2,
        "base_mass": 65.0,
        "base_height": 1.7,
        "frame_factor": 1.0,
    }

    base_brain = {
        "W1": np.zeros((Humlet.N_HIDDEN, Humlet.N_INPUTS)),
        "b1": np.zeros(Humlet.N_HIDDEN),
        "W2": np.zeros((Humlet.N_OUTPUTS, Humlet.N_HIDDEN)),
        "b2": np.zeros(Humlet.N_OUTPUTS),
    }

    awake_brain = copy.deepcopy(base_brain)
    awake_brain["b2"][0] = 1.0  # bias movement
    awake_brain["b2"][4] = -4.0  # suppress rest

    rest_brain = copy.deepcopy(base_brain)
    rest_brain["b2"][0] = 1.0  # bias movement equally
    rest_brain["b2"][4] = 4.0  # encourage rest

    awake = Humlet(env, genome=genome, brain=awake_brain, seed=21)
    resting = Humlet(env, genome=genome, brain=rest_brain, seed=22)

    for h in (awake, resting):
        h.energy = 800.0
        h.health = h.max_health

    awake.update(env, [awake], [], max_population=6)
    resting.update(env, [resting], [], max_population=6)

    awake_speed = math.hypot(awake.vx, awake.vy)
    resting_speed = math.hypot(resting.vx, resting.vy)

    assert resting_speed < awake_speed * 0.7
    assert resting.energy > awake.energy
    assert resting.rest_intensity > awake.rest_intensity
