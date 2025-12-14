import math

import pytest

pytest.importorskip("numpy")
pytest.importorskip("pygame")

import numpy as np

from humlet_simulation.environment import Environment, Food
from humlet_simulation.humlet import Humlet
from humlet_simulation.simulation import Simulation
from humlet_simulation.stats import RegionTraitStats


def test_simulation_initialises_components_and_stats(headless_pygame):
    sim = Simulation(
        world_width=180,
        world_height=120,
        num_humlets=4,
        panel_width=60,
        margin=5,
        seed=1234,
    )

    assert sim.env.width == 180
    assert sim.env.height == 120
    assert len(sim.humlets) == 4
    assert set(sim.agent_seeds.keys()) == {h.id for h in sim.humlets}
    assert isinstance(sim.region_stats, RegionTraitStats)
    assert sim.region_stats.cols == sim.env.cols
    assert any(isinstance(obj, Food) for obj in sim.env.objects)

    sim.stats.update(sim.tick, sim.humlets, sim.env)
    assert sim.stats.latest is not None


def test_simulation_remains_stable_over_many_ticks(headless_pygame):
    sim = Simulation(
        world_width=160,
        world_height=120,
        num_humlets=8,
        panel_width=30,
        margin=5,
        seed=4321,
    )

    for _ in range(1500):
        sim._update_simulation_step()

    assert sim.tick == 1500
    assert len(sim.humlets) >= 0

    for h in sim.humlets:
        assert math.isfinite(h.energy) and 0.0 <= h.energy <= h.max_energy
        assert math.isfinite(h.health) and 0.0 <= h.health <= h.max_health

        for key, value in h.genome.items():
            assert math.isfinite(value)
            if key == "metabolism_rate":
                assert 0.01 <= value <= 0.08
            if key == "speed_trait":
                assert 0.6 <= value <= 1.8
            if key == "sense_range":
                assert 40.0 <= value <= 320.0
            if key == "aggression" or key == "sociability" or key == "curiosity_trait":
                assert 0.0 <= value <= 1.0
            if key == "base_mass":
                assert 30.0 <= value <= 150.0
            if key == "base_height":
                assert 1.0 <= value <= 2.5
            if key == "frame_factor":
                assert 0.5 <= value <= 1.5
            if key == "lifespan":
                assert value >= 800

    assert len(sim.humlets) <= sim.max_population
    assert sim.stats.latest is not None


def test_simulation_handles_extinction_without_errors(headless_pygame):
    sim = Simulation(
        world_width=120,
        world_height=90,
        num_humlets=0,
        panel_width=30,
        margin=5,
        seed=777,
    )

    for _ in range(10):
        sim._update_simulation_step()

    assert sim.humlets == []
    assert sim.stats.latest is not None
    assert sim.stats.latest.population == 0


def test_toroidal_distances_are_symmetric_and_wrap(headless_pygame):
    env = Environment(100, 100)
    h1 = Humlet(env, seed=1)
    h2 = Humlet(env, seed=2)

    h1.x, h1.y = 2.0, 50.0
    h2.x, h2.y = 98.0, 50.0

    dx1, dy1 = h1._wrapped_delta(env, h2.x, h2.y)
    dx2, dy2 = h2._wrapped_delta(env, h1.x, h1.y)

    assert math.isclose(dx1, -dx2, abs_tol=1e-6)
    assert math.isclose(dy1, -dy2, abs_tol=1e-6)
    assert math.isclose(math.hypot(dx1, dy1), math.hypot(dx2, dy2), rel_tol=1e-6)

    food_left = Food(98.0, 50.0, nutrition=10.0)
    food_right = Food(2.0, 50.0, nutrition=10.0)
    env.add_object(food_left)
    env.add_object(food_right)

    h1.smell.range = 50.0
    h2.smell.range = 50.0

    dir_h1 = h1.smell.sense(env)
    dir_h2 = h2.smell.sense(env)

    assert dir_h1[0] < -0.5 and abs(dir_h1[1]) < 0.2
    assert dir_h2[0] > 0.5 and abs(dir_h2[1]) < 0.2


def test_brain_outputs_remain_bounded_and_need_sensitive():
    env = Environment(60, 60)

    custom_brain = {
        "W1": np.zeros((Humlet.N_HIDDEN, Humlet.N_INPUTS)),
        "b1": np.zeros(Humlet.N_HIDDEN),
        "W2": np.zeros((Humlet.N_OUTPUTS, Humlet.N_HIDDEN)),
        "b2": np.zeros(Humlet.N_OUTPUTS),
    }

    custom_brain["W1"][0, 0] = 3.0  # hunger -> hidden
    custom_brain["W1"][1, 3] = 3.0  # energy_norm -> hidden
    custom_brain["W2"][0, 1] = 1.5  # movement uses energy signal
    custom_brain["W2"][1, 1] = 0.5
    custom_brain["W2"][2, 0] = 2.5  # eat responds to hunger
    custom_brain["W2"][3, 0] = -2.0  # reproduce suppressed by hunger

    h = Humlet(env, brain=custom_brain)

    hungry_inputs = np.zeros(Humlet.N_INPUTS)
    hungry_inputs[0] = 1.0  # hunger_need
    hungry_inputs[3] = 0.2  # energy_norm

    full_inputs = np.zeros(Humlet.N_INPUTS)
    full_inputs[0] = 0.0
    full_inputs[3] = 1.0

    hungry_out = h._brain_forward(hungry_inputs)
    full_out = h._brain_forward(full_inputs)

    assert all(math.isfinite(v) for v in hungry_out)
    assert all(math.isfinite(v) for v in full_out)

    for move_x in hungry_out[:2].tolist() + full_out[:2].tolist():
        assert -1.0 <= move_x <= 1.0

    for action in hungry_out[2:].tolist() + full_out[2:].tolist():
        assert 0.0 <= action <= 1.0

    assert hungry_out[2] > hungry_out[3]
    assert math.hypot(*hungry_out[:2]) < math.hypot(*full_out[:2])

    varied_outputs = []
    for hunger in (0.0, 0.5, 1.0):
        inputs = np.zeros(Humlet.N_INPUTS)
        inputs[0] = hunger
        inputs[3] = 0.6
        varied_outputs.append(h._brain_forward(inputs))

    eat_values = [o[2] for o in varied_outputs]
    assert max(eat_values) - min(eat_values) > 0.05
