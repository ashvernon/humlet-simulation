from __future__ import annotations

import datetime as _dt
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class DeathEvent:
    tick: int
    humlet_id: int
    family_id: int
    generation: int
    age: int
    x: float
    y: float
    region_col: int | None
    region_row: int | None
    energy: float
    health: float
    hunger_need: float
    safety_need: float
    social_need: float
    cause: str
    last_action: str | None = None
    brain_outputs: str | None = None

    def as_row(self) -> tuple:
        return (
            self.tick,
            self.humlet_id,
            self.family_id,
            self.generation,
            self.age,
            self.x,
            self.y,
            self.region_col,
            self.region_row,
            self.energy,
            self.health,
            self.hunger_need,
            self.safety_need,
            self.social_need,
            self.cause,
            self.last_action,
            self.brain_outputs,
        )


def _percentiles(data: list[float]) -> tuple[float, float, float]:
    if not data:
        return 0.0, 0.0, 0.0
    arr = np.array(sorted(data))
    return float(np.percentile(arr, 10)), float(np.percentile(arr, 50)), float(np.percentile(arr, 90))


class TelemetryRecorder:
    """Lightweight SQLite-backed telemetry pipeline."""

    def __init__(
        self,
        run_id: str,
        *,
        base_seed: int,
        world_size: tuple[int, int],
        snapshot_interval: int = 50,
        base_path: str | Path = "reports",
    ) -> None:
        self.run_id = run_id
        self.snapshot_interval = snapshot_interval
        self.base_seed = base_seed
        self.world_size = world_size

        self.base_path = Path(base_path)
        self.run_dir = self.base_path / f"run_{self.run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.run_dir / f"run_{self.run_id}.sqlite"

        self._conn = sqlite3.connect(self.db_path)
        self._init_db()

    # ------------------------------------------------------------------ #
    # Database schema
    # ------------------------------------------------------------------ #
    def _init_db(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS run_meta (
                run_id TEXT PRIMARY KEY,
                seed INTEGER,
                world_width INTEGER,
                world_height INTEGER,
                start_time TEXT
            )
            """
        )
        cur.execute(
            """
            INSERT OR REPLACE INTO run_meta(run_id, seed, world_width, world_height, start_time)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                self.run_id,
                self.base_seed,
                self.world_size[0],
                self.world_size[1],
                _dt.datetime.utcnow().isoformat(),
            ),
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS snapshots (
                tick INTEGER PRIMARY KEY,
                population INTEGER,
                births INTEGER,
                deaths INTEGER,
                avg_age REAL,
                median_age REAL,
                max_age REAL,
                avg_energy REAL,
                median_energy REAL,
                avg_health REAL,
                median_health REAL,
                avg_speed REAL,
                avg_metabolism REAL,
                avg_sense_range REAL,
                avg_aggression REAL,
                avg_sociability REAL,
                avg_curiosity REAL,
                food_count INTEGER,
                stone_count INTEGER,
                tree_count INTEGER,
                population_grid TEXT,
                food_grid TEXT
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS deaths (
                tick INTEGER,
                humlet_id INTEGER,
                family_id INTEGER,
                generation INTEGER,
                age INTEGER,
                x REAL,
                y REAL,
                region_col INTEGER,
                region_row INTEGER,
                energy REAL,
                health REAL,
                hunger_need REAL,
                safety_need REAL,
                social_need REAL,
                cause TEXT,
                last_action TEXT,
                brain_outputs TEXT
            )
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #
    def log_deaths(self, events: Iterable[DeathEvent]) -> None:
        events = list(events)
        if not events:
            return
        cur = self._conn.cursor()
        cur.executemany(
            """
            INSERT INTO deaths(
                tick, humlet_id, family_id, generation, age, x, y, region_col, region_row,
                energy, health, hunger_need, safety_need, social_need, cause, last_action, brain_outputs
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [e.as_row() for e in events],
        )
        self._conn.commit()

    def record_snapshot(
        self,
        *,
        tick: int,
        humlets,
        env,
        births: int,
        deaths: int,
    ) -> None:
        alive = [h for h in humlets if getattr(h, "alive", False)]
        ages = [h.age for h in alive]
        energies = [h.energy for h in alive]
        healths = [h.health for h in alive]

        def avg(values: Iterable[float]) -> float:
            vals = list(values)
            return float(sum(vals) / len(vals)) if vals else 0.0

        _, med_energy, _ = _percentiles(energies)
        _, med_health, _ = _percentiles(healths)
        _, med_age, max_age = (0.0, 0.0, 0.0)
        if ages:
            _, med_age, _ = _percentiles(ages)
            max_age = max(ages)

        avg_speed = avg(h.speed_trait for h in alive)
        avg_metabolism = avg(h.metabolism_rate for h in alive)
        avg_sense_range = avg(h.sense_range for h in alive)
        avg_aggression = avg(h.aggression for h in alive)
        avg_sociability = avg(h.sociability for h in alive)
        avg_curiosity = avg(h.curiosity_trait for h in alive)

        food_count = sum(1 for obj in env.objects if getattr(obj, "type", "") == "food")
        stone_count = sum(1 for obj in env.objects if getattr(obj, "type", "") == "stone")
        tree_count = sum(1 for obj in env.objects if getattr(obj, "type", "") == "tree")

        pop_grid = self._population_grid(alive, env)
        food_grid = self._food_grid(env)

        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO snapshots(
                tick, population, births, deaths, avg_age, median_age, max_age,
                avg_energy, median_energy, avg_health, median_health,
                avg_speed, avg_metabolism, avg_sense_range, avg_aggression, avg_sociability, avg_curiosity,
                food_count, stone_count, tree_count, population_grid, food_grid
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tick,
                len(alive),
                births,
                deaths,
                avg(ages),
                med_age,
                max_age,
                avg(energies),
                med_energy,
                avg(healths),
                med_health,
                avg_speed,
                avg_metabolism,
                avg_sense_range,
                avg_aggression,
                avg_sociability,
                avg_curiosity,
                food_count,
                stone_count,
                tree_count,
                json.dumps(pop_grid),
                json.dumps(food_grid),
            ),
        )
        self._conn.commit()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _population_grid(self, humlets, env) -> list[list[int]]:
        if not env.regions:
            return []
        grid = [[0 for _ in range(env.cols)] for _ in range(env.rows)]
        for h in humlets:
            region = env.get_region_at(h.x, h.y)
            if region is None:
                continue
            grid[region.row][region.col] += 1
        return grid

    def _food_grid(self, env) -> list[list[int]]:
        if not env.regions:
            return []
        grid = [[0 for _ in range(env.cols)] for _ in range(env.rows)]
        for obj in env.objects:
            if getattr(obj, "type", "") != "food":
                continue
            region = env.get_region_at(obj.x, obj.y)
            if region is None:
                continue
            grid[region.row][region.col] += 1
        return grid

    def close(self) -> None:
        self._conn.close()


__all__ = ["TelemetryRecorder", "DeathEvent"]
