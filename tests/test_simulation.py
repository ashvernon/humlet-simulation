import pytest

pytest.importorskip("numpy")
pytest.importorskip("pygame")

from humlet_simulation.environment import Food
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
