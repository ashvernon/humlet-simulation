import math

import pytest

pytest.importorskip("numpy")

from humlet_simulation.environment import Environment, Food, StoneDeposit, Tree
from humlet_simulation.humlet import Humlet


def test_environment_generates_region_grid_and_resources():
    env = Environment(240, 200)

    assert len(env.regions) == env.rows
    assert all(len(row) == env.cols for row in env.regions)

    static_resources = [o for o in env.objects if isinstance(o, (Tree, StoneDeposit))]
    assert static_resources, "Expected initial trees or stone deposits to be spawned"

    assert math.isclose(env.tile_w, env.width / env.cols)
    assert math.isclose(env.tile_h, env.height / env.rows)


def test_smell_uses_toroidal_wraparound():
    env = Environment(100, 100)
    humlet = Humlet(env, seed=42)

    humlet.x = 95.0
    humlet.y = 50.0
    humlet.smell.range = 150.0

    env.add_object(Food(5.0, 50.0, nutrition=10.0))

    dx, dy = humlet.smell.sense(env)

    assert dx > 0.5, "Wrapped scent should point across the boundary toward the food"
    assert abs(dy) < 0.1
