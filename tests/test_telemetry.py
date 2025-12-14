import sqlite3
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from humlet_simulation.environment import Environment
from humlet_simulation.humlet import Humlet
from humlet_simulation.telemetry import DeathEvent, TelemetryRecorder


def _make_recorder(tmp_path):
    return TelemetryRecorder(
        "test_run",
        base_seed=123,
        world_size=(50, 50),
        snapshot_interval=1,
        base_path=tmp_path,
    )


def test_death_logging_records_cause(tmp_path):
    env = Environment(50, 50)
    humlet = Humlet(env, seed=1)
    humlet.energy = 0.0  # trigger starvation path

    recorder = _make_recorder(tmp_path)

    humlet.update(env, [humlet], [], max_population=5, spatial_index=None)
    assert not humlet.alive
    assert humlet.death_info is not None

    death = DeathEvent(
        tick=1,
        humlet_id=humlet.id,
        family_id=humlet.family_id,
        generation=humlet.generation,
        age=humlet.age,
        x=humlet.x,
        y=humlet.y,
        region_col=humlet.death_info.get("region_col"),
        region_row=humlet.death_info.get("region_row"),
        energy=humlet.energy,
        health=humlet.health,
        hunger_need=humlet.death_info.get("hunger_need", humlet.hunger_need),
        safety_need=humlet.death_info.get("safety_need", humlet.safety_need),
        social_need=humlet.death_info.get("social_need", humlet.social_need),
        cause=humlet.death_info.get("cause", "other"),
        last_action=humlet.death_info.get("last_action"),
        brain_outputs=humlet.death_info.get("brain_outputs"),
    )
    recorder.log_deaths([death])

    conn = sqlite3.connect(recorder.db_path)
    cause = conn.execute("SELECT cause FROM deaths LIMIT 1").fetchone()[0]
    conn.close()
    assert cause == "starvation"


def test_snapshot_records_population_and_resources(tmp_path):
    env = Environment(50, 50)
    humlets = [Humlet(env, seed=seed) for seed in (2, 3, 4)]

    recorder = _make_recorder(tmp_path)
    recorder.record_snapshot(
        tick=10,
        humlets=humlets,
        env=env,
        births=2,
        deaths=1,
    )

    conn = sqlite3.connect(recorder.db_path)
    row = conn.execute(
        "SELECT population, births, deaths FROM snapshots WHERE tick=10"
    ).fetchone()
    conn.close()

    assert row == (3, 2, 1)
