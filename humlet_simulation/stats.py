# humlet_simulation/stats.py
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List

import math

from .humlet import Humlet
from .environment import Environment, Food


# ----------------------------------------------------------------------
# Global population snapshot
# ----------------------------------------------------------------------
@dataclass
class StatsSnapshot:
    tick: int
    population: int

    # Reproduction state
    pregnant_count: int
    avg_gestation_progress: float

    # Core trait / physiology aggregates
    avg_speed: float
    avg_sense_range: float
    avg_aggression: float
    avg_sociability: float
    avg_metabolism_rate: float
    avg_age: float
    avg_energy: float
    avg_health: float
    avg_stomach_fill: float
    avg_digestion_flow: float
    avg_absorption_efficiency: float

    # New: physical body traits / growth
    avg_mass: float
    avg_height: float
    avg_frame_factor: float

    # New: carrying / proto-economy at individual level
    avg_carry_wood: float
    avg_carry_stone: float

    # New: spatial behaviour – home-range usage
    avg_home_distance: float

    # Evolution / lineage
    max_generation: int
    avg_generation: float
    num_families: int

    # Higher-level / Maslow-ish & social structure
    avg_esteem_level: float
    avg_curiosity_trait: float
    avg_curiosity_drive: float
    avg_offspring_count: float
    max_offspring_count: int
    avg_neighbors: float

    # Early Civilisation / Village economy
    village_food: float
    village_wood: float
    village_stone: float
    avg_food_per_capita: float
    avg_wood_per_capita: float
    avg_stone_per_capita: float

    # Environment / ecology outputs
    wild_food_count: int
    food_capacity: int
    food_energy_pool: float
    productivity_per_tick: float
    light_level: float
    day_progress: float
    season_progress: float


# ----------------------------------------------------------------------
# Global evolution stats over time
# ----------------------------------------------------------------------
@dataclass
class EvolutionStats:
    """
    Tracks how 'evolved' the population is over time:
    - trait averages (speed, sense, aggression, sociability, metabolism)
    - physiological state (age, energy, health, body mass/height)
    - lineage (generations, families)
    - higher-level behaviour (esteem, curiosity, social clustering)
    - spatial behaviour (distance from birthplace)
    - early civilisation development (village resource accumulation)
    """
    history: list[StatsSnapshot] = field(default_factory=list)
    max_history_len: int = 500

    latest: StatsSnapshot | None = None

    def latest_as_dict(self) -> dict | None:
        if self.latest is None:
            return None
        return asdict(self.latest)

    def history_as_dicts(self) -> list[dict]:
        return [asdict(s) for s in self.history]

    def update(self, tick: int, humlets: List[Humlet], env: Environment | None = None) -> None:
        alive = [h for h in humlets if h.alive]
        n = len(alive)

        # Default village stats if env not passed
        v_food = v_wood = v_stone = 0.0
        food_capacity = 0
        food_energy_pool = 0.0
        productivity_per_tick = 0.0
        light_level = 0.0
        day_progress = 0.0
        season_progress = 0.0
        wild_food_count = 0

        if env is not None and hasattr(env, "village"):
            totals = env.village.totals()
            v_food = totals.get("food", 0.0)
            v_wood = totals.get("wood", 0.0)
            v_stone = totals.get("stone", 0.0)
            food_capacity = getattr(env, "food_capacity", 0)
            food_energy_pool = getattr(env, "food_energy_pool", 0.0)
            productivity_per_tick = getattr(env, "_productivity_per_tick", 0.0)
            light_level = getattr(env, "light_level", 0.0)
            day_length = getattr(env, "day_length", 1) or 1
            season_length = getattr(env, "season_length", 1) or 1
            day_progress = (getattr(env, "time", 0) % day_length) / day_length
            season_progress = (getattr(env, "time", 0) % season_length) / season_length
            wild_food_count = sum(1 for o in env.objects if isinstance(o, Food))

        if n == 0:
            snapshot = StatsSnapshot(
                tick=tick,
                population=0,
                pregnant_count=0,
                avg_gestation_progress=0.0,
                avg_speed=0.0,
                avg_sense_range=0.0,
                avg_aggression=0.0,
                avg_sociability=0.0,
                avg_metabolism_rate=0.0,
                avg_age=0.0,
                avg_energy=0.0,
                avg_health=0.0,
                avg_stomach_fill=0.0,
                avg_digestion_flow=0.0,
                avg_absorption_efficiency=0.0,
                # physical
                avg_mass=0.0,
                avg_height=0.0,
                avg_frame_factor=0.0,
                # carrying
                avg_carry_wood=0.0,
                avg_carry_stone=0.0,
                # space use
                avg_home_distance=0.0,
                # lineage
                max_generation=0,
                avg_generation=0.0,
                num_families=0,
                # higher-level
                avg_esteem_level=0.0,
                avg_curiosity_trait=0.0,
                avg_curiosity_drive=0.0,
                avg_offspring_count=0.0,
                max_offspring_count=0,
                avg_neighbors=0.0,
                # Civilisation metrics
                village_food=v_food,
                village_wood=v_wood,
                village_stone=v_stone,
                avg_food_per_capita=0.0,
                avg_wood_per_capita=0.0,
                avg_stone_per_capita=0.0,
                # ecology
                wild_food_count=wild_food_count,
                food_capacity=food_capacity,
                food_energy_pool=food_energy_pool,
                productivity_per_tick=productivity_per_tick,
                light_level=light_level,
                day_progress=day_progress,
                season_progress=season_progress,
            )
        else:
            def avg(fn): return sum(fn(h) for h in alive) / n

            # Reproduction state
            pregnant_humlets = [h for h in alive if getattr(h, "pregnant", False)]
            pregnant_count = len(pregnant_humlets)

            def gestation_progress(h: Humlet) -> float:
                if getattr(h, "gestation_period", 0) <= 0:
                    return 0.0
                remaining = getattr(h, "gestation_timer", 0)
                period = getattr(h, "gestation_period", 1)
                return max(0.0, min(1.0, 1.0 - (remaining / max(1, period))))

            avg_gestation_progress = (
                sum(gestation_progress(h) for h in pregnant_humlets) / pregnant_count
                if pregnant_count > 0 else 0.0
            )

            # Core traits
            avg_speed = avg(lambda h: h.speed_trait)
            avg_sense_range = avg(lambda h: h.sense_range)
            avg_aggression = avg(lambda h: h.aggression)
            avg_sociability = avg(lambda h: h.sociability)
            avg_metabolism_rate = avg(lambda h: h.metabolism_rate)

            # Physiological state
            avg_age = avg(lambda h: h.age)
            avg_energy = avg(lambda h: h.energy)
            avg_health = avg(lambda h: h.health)
            avg_stomach_fill = avg(
                lambda h: (
                    getattr(h, "stomach_content", 0.0)
                    / getattr(h, "stomach_capacity", 1.0)
                    if getattr(h, "stomach_capacity", 0.0) > 0
                    else 0.0
                )
            )
            avg_digestion_flow = avg(lambda h: getattr(h, "digestion_flow", 0.0))
            avg_absorption_efficiency = avg(lambda h: getattr(h, "absorption_efficiency", 0.0))

            # Physical body
            avg_mass = avg(lambda h: getattr(h, "mass", 0.0))
            avg_height = avg(lambda h: getattr(h, "height", 0.0))
            avg_frame_factor = avg(lambda h: getattr(h, "frame_factor", 1.0))

            # Carrying
            avg_carry_wood = avg(lambda h: getattr(h, "carry_wood", 0.0))
            avg_carry_stone = avg(lambda h: getattr(h, "carry_stone", 0.0))

            # Distance from home (home-range usage)
            def home_dist(h: Humlet) -> float:
                dx = h.x - getattr(h, "home_x", h.x)
                dy = h.y - getattr(h, "home_y", h.y)
                return math.hypot(dx, dy)

            avg_home_distance = avg(home_dist)

            # Lineage
            max_generation = max(h.generation for h in alive)
            avg_generation = avg(lambda h: h.generation)
            families = {h.family_id for h in alive}

            # Higher-level motivations & social
            avg_esteem_level = avg(lambda h: getattr(h, "esteem_level", 0.0))
            avg_curiosity_trait = avg(lambda h: getattr(h, "curiosity_trait", 0.0))
            avg_curiosity_drive = avg(lambda h: getattr(h, "curiosity_drive", 0.0))
            avg_offspring_count = avg(lambda h: getattr(h, "offspring_count", 0))
            max_offspring_count = max(getattr(h, "offspring_count", 0) for h in alive)
            avg_neighbors = avg(lambda h: getattr(h, "neighbor_count", 0))

            # Per-capita village resources (early civilisation metric)
            avg_food_per_capita = v_food / n
            avg_wood_per_capita = v_wood / n
            avg_stone_per_capita = v_stone / n

            snapshot = StatsSnapshot(
                tick=tick,
                population=n,
                pregnant_count=pregnant_count,
                avg_gestation_progress=avg_gestation_progress,
                avg_speed=avg_speed,
                avg_sense_range=avg_sense_range,
                avg_aggression=avg_aggression,
                avg_sociability=avg_sociability,
                avg_metabolism_rate=avg_metabolism_rate,
                avg_age=avg_age,
                avg_energy=avg_energy,
                avg_health=avg_health,
                avg_stomach_fill=avg_stomach_fill,
                avg_digestion_flow=avg_digestion_flow,
                avg_absorption_efficiency=avg_absorption_efficiency,
                # physical
                avg_mass=avg_mass,
                avg_height=avg_height,
                avg_frame_factor=avg_frame_factor,
                # carrying
                avg_carry_wood=avg_carry_wood,
                avg_carry_stone=avg_carry_stone,
                # space use
                avg_home_distance=avg_home_distance,
                # lineage
                max_generation=max_generation,
                avg_generation=avg_generation,
                num_families=len(families),
                # higher-level
                avg_esteem_level=avg_esteem_level,
                avg_curiosity_trait=avg_curiosity_trait,
                avg_curiosity_drive=avg_curiosity_drive,
                avg_offspring_count=avg_offspring_count,
                max_offspring_count=max_offspring_count,
                avg_neighbors=avg_neighbors,
                # Civilisation metrics
                village_food=v_food,
                village_wood=v_wood,
                village_stone=v_stone,
                avg_food_per_capita=avg_food_per_capita,
                avg_wood_per_capita=avg_wood_per_capita,
                avg_stone_per_capita=avg_stone_per_capita,
                # ecology
                wild_food_count=wild_food_count,
                food_capacity=food_capacity,
                food_energy_pool=food_energy_pool,
                productivity_per_tick=productivity_per_tick,
                light_level=light_level,
                day_progress=day_progress,
                season_progress=season_progress,
            )

        self.latest = snapshot
        self.history.append(snapshot)
        if len(self.history) > self.max_history_len:
            self.history.pop(0)


# ----------------------------------------------------------------------
# Per-region trait statistics for genetic drift heatmaps
# ----------------------------------------------------------------------
class RegionTraitStats:
    """
    Tracks average traits per environment region (tile), so we can render
    heatmaps showing how different zones evolve different trait profiles.
    """

    def __init__(self, env: Environment):
        self.env = env
        self.cols = env.cols
        self.rows = env.rows
        self.reset()

    def reset(self) -> None:
        # Each region stores sums + count; means are computed later
        self.regions = [
            [dict(
                count=0,
                met=0.0,
                spd=0.0,
                sns=0.0,
                soc=0.0,
                agg=0.0,
                m=0.0,
                h=0.0,
                met_mean=0.0,
                spd_mean=0.0,
                sns_mean=0.0,
                soc_mean=0.0,
                agg_mean=0.0,
                m_mean=0.0,
                h_mean=0.0,
            ) for _c in range(self.cols)]
            for _r in range(self.rows)
        ]

    def accumulate(self, h: Humlet) -> None:
        # Only care about alive humlets
        if not h.alive:
            return

        col = int(h.x // self.env.tile_w)
        row = int(h.y // self.env.tile_h)

        if 0 <= col < self.cols and 0 <= row < self.rows:
            reg = self.regions[row][col]
            reg["count"] += 1
            reg["met"] += h.metabolism_rate
            reg["spd"] += h.speed_trait
            reg["sns"] += h.sense_range
            reg["soc"] += h.sociability
            reg["agg"] += h.aggression
            reg["m"] += getattr(h, "mass", 0.0)
            reg["h"] += getattr(h, "height", 0.0)

    def compute_means(self) -> None:
        for r in range(self.rows):
            for c in range(self.cols):
                reg = self.regions[r][c]
                if reg["count"] > 0:
                    inv = 1.0 / reg["count"]
                    reg["met_mean"] = reg["met"] * inv
                    reg["spd_mean"] = reg["spd"] * inv
                    reg["sns_mean"] = reg["sns"] * inv
                    reg["soc_mean"] = reg["soc"] * inv
                    reg["agg_mean"] = reg["agg"] * inv
                    reg["m_mean"] = reg["m"] * inv
                    reg["h_mean"] = reg["h"] * inv
                else:
                    reg["met_mean"] = reg["spd_mean"] = 0.0
                    reg["sns_mean"] = reg["soc_mean"] = 0.0
                    reg["agg_mean"] = 0.0
                    reg["m_mean"] = reg["h_mean"] = 0.0
