from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Type

from .spatial_hash import SpatialHash
from .village import Village


# ------------------------------------------------------------------ #
# World objects
# ------------------------------------------------------------------ #
class WorldObject:
    """Base class for objects in the world."""

    def __init__(self, x: float, y: float, *, radius: float = 6.0, solid: bool = False):
        self.x = x
        self.y = y
        self.pickable = False
        self.type = "object"
        self.radius = radius
        self.solid = solid


class Food(WorldObject):
    """Food resource that humlets can consume to gain energy."""

    def __init__(self, x: float, y: float, nutrition: float = 200.0):
        super().__init__(x, y, radius=4.0, solid=False)
        self.type = "food"
        self.nutrition = nutrition
        self.pickable = False  # eaten in place


class Rock(WorldObject):
    """A rock that can be picked up and used as a primitive tool/weapon."""

    def __init__(self, x: float, y: float):
        super().__init__(x, y, radius=6.0, solid=True)
        self.type = "rock"
        self.pickable = True


class Shelter(WorldObject):
    """A shelter that reduces safety pressure and energy loss."""

    def __init__(self, x: float, y: float, capacity: int = 5):
        super().__init__(x, y, radius=12.0, solid=True)
        self.type = "shelter"
        self.capacity = capacity
        self.pickable = False


class Tree(WorldObject):
    """Tree that can be harvested for wood."""

    def __init__(self, x: float, y: float, wood_amount: float = 20.0):
        super().__init__(x, y, radius=10.0, solid=True)
        self.type = "tree"
        self.wood_amount = wood_amount
        self.pickable = False  # harvested in place


class StoneDeposit(WorldObject):
    """Stone deposit that can be mined for stone."""

    def __init__(self, x: float, y: float, stone_amount: float = 20.0):
        super().__init__(x, y, radius=10.0, solid=True)
        self.type = "stone"
        self.stone_amount = stone_amount
        self.pickable = False


# ------------------------------------------------------------------ #
# Biome / region grid
# ------------------------------------------------------------------ #
@dataclass
class Region:
    """A coarse tile of the world with its own 'local' environment."""

    col: int
    row: int
    biome: str  # "water", "grassland", "forest", "desert", "mountain"
    temp_offset: float  # °C offset relative to global temperature
    humidity: float  # 0–1
    fertility: float  # 0–1 (how likely food is to spawn here)
    roughness: float  # 0–1 (movement difficulty, if we use it later)
    water: bool  # convenience flag


# ------------------------------------------------------------------ #
# Environment
# ------------------------------------------------------------------ #
class Environment:
    """Global environment + coarse biome grid + world objects."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

        # Simple single-village world for now (centre of map)
        self.village = Village(width * 0.5, height * 0.5)

        # Global environmental factors
        self.temperature = 20.0
        self.air_quality = 1.0
        self.air_density = 1.0

        # Time & cycles
        self.time = 0
        self.season_length = 80000
        self.day_length = 800
        self.light_level = 1.0

        # Objects in world
        self.objects: List[WorldObject] = []
        self.object_index = SpatialHash(self.width, self.height, cell_size=64.0)

        # Food dynamics (biomass-conserving)
        self.food_respawn_interval = 20
        self._last_food_spawn = 0
        self.base_food_nutrition = 80.0
        self.food_energy_pool = 0.0
        self.food_capacity = 0
        self._productivity_per_tick = 0.0
        self._base_productivity_per_tick = 0.0
        self.avg_humidity = 0.5
        self.land_fraction = 1.0
        self.land_area = float(self.width * self.height)

        # ---------------------------------------------------------- #
        # Biome grid setup (Earth-like latitude + noise)
        # ---------------------------------------------------------- #
        # Initial desired grid size; _generate_regions will normalise
        self.cols = 32
        self.rows = 32
        self.tile_w = self.width / self.cols
        self.tile_h = self.height / self.rows

        self.regions: list[list[Region]] = []
        self._generate_regions()

        # Food carrying capacity and productivity depend on the land surface
        self._initialise_food_budget()

        # --- Static resource nodes, now biome-aware ---
        self._spawn_initial_resources()

    # ------------------------------------------------------------------ #
    # Initial static resources (trees & stone)
    # ------------------------------------------------------------------ #
    def _spawn_initial_resources(self) -> None:
        """Scatter trees and stone deposits in a clustered, biome-aware way."""

        def is_near_water(reg: Region) -> bool:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr = reg.row + dr
                    cc = reg.col + dc
                    if 0 <= rr < self.rows and 0 <= cc < self.cols:
                        if self.regions[rr][cc].water:
                            return True
            return False

        tree_candidates: list[Region] = []
        stone_candidates: list[Region] = []

        for row in self.regions:
            for reg in row:
                if reg.water:
                    continue

                near_water = is_near_water(reg)

                # ---- Trees: like fertile, humid, forest, near water ----
                tree_weight = 0.5 + 4.0 * reg.fertility + 2.0 * reg.humidity
                if reg.biome == "forest":
                    tree_weight *= 1.6
                if near_water:
                    tree_weight *= 1.7

                if tree_weight > 0:
                    tree_candidates.extend([reg] * max(1, int(tree_weight)))

                # ---- Stone: like mountains / rough terrain ----
                if reg.biome == "mountain":
                    stone_weight = 6.0
                elif reg.biome == "desert":
                    stone_weight = 3.0
                else:
                    stone_weight = 0.5 + 3.0 * reg.roughness

                if stone_weight > 0:
                    stone_candidates.extend([reg] * max(1, int(stone_weight)))

        def spawn_clusters(
            candidates: list[Region],
            n_objects: int,
            radius_factor: float,
            cls,
        ) -> None:
            if not candidates:
                return

            remaining = n_objects
            while remaining > 0:
                centre = random.choice(candidates)
                cluster_size = random.randint(2, 5)
                cx = (centre.col + 0.5) * self.tile_w
                cy = (centre.row + 0.5) * self.tile_h

                for _ in range(cluster_size):
                    if remaining <= 0:
                        break

                    dx = random.uniform(-radius_factor, radius_factor) * self.tile_w
                    dy = random.uniform(-radius_factor, radius_factor) * self.tile_h
                    x = max(0.0, min(self.width - 1.0, cx + dx))
                    y = max(0.0, min(self.height - 1.0, cy + dy))

                    if cls is Tree:
                        self.objects.append(Tree(x, y))
                    else:
                        self.objects.append(StoneDeposit(x, y))

                    remaining -= 1

        # Roughly match your previous counts
        spawn_clusters(tree_candidates, n_objects=40, radius_factor=0.7, cls=Tree)
        spawn_clusters(stone_candidates, n_objects=25, radius_factor=0.5, cls=StoneDeposit)

    # ------------------------------------------------------------------ #
    # Biome generation
    # ------------------------------------------------------------------ #
    def _generate_regions(self) -> None:
        """Generate a simple 'planet' with latitude bands and noisy biomes."""

        regions: list[list[Region]] = []

        for row in range(self.rows):
            lat = row / max(1, self.rows - 1)  # 0 = top pole, 1 = bottom pole
            # Warmest near equator, cooler near poles
            lat_temp = 1.0 - abs(lat - 0.5) * 2.0
            lat_temp = max(0.0, min(1.0, lat_temp))

            # Map to a temp offset range, e.g. poles -7°C, equator +7°C
            base_temp_offset = (lat_temp - 0.5) * 14.0  # -7 .. +7

            row_regions: list[Region] = []
            for col in range(self.cols):
                x_frac = col / max(1, self.cols - 1)
                noise = random.uniform(-1.0, 1.0)

                # Probability of water: more at poles + some randomness
                water_prob = 0.15 + 0.25 * (1.0 - lat_temp)  # more polar seas
                water_prob += 0.1 * (0.5 - abs(x_frac - 0.5))  # a bit more near centre
                water_prob = max(0.05, min(0.7, water_prob))

                is_water = random.random() < water_prob and noise < 0.2

                if is_water:
                    biome = "water"
                    temp_offset = base_temp_offset - 10.0  # water slightly cooler # adjust to 10
                    humidity = 0.9
                    fertility = 0.15
                    roughness = 0.1
                else:
                    # Land biomes decided by latitude + noise
                    if lat_temp > 0.65:
                        # Equatorial band: forest / grassland / a bit of desert
                        if noise > 0.4:
                            biome = "forest"
                            fertility = 0.9
                            humidity = 0.8
                            roughness = 0.4
                        elif noise < -0.4:
                            biome = "desert"
                            fertility = 0.2
                            humidity = 0.2
                            roughness = 0.2
                        else:
                            biome = "grassland"
                            fertility = 0.7
                            humidity = 0.5
                            roughness = 0.2
                    elif lat_temp > 0.3:
                        # Temperate band: mostly grassland/forest
                        if noise > 0.5:
                            biome = "forest"
                            fertility = 0.8
                            humidity = 0.7
                            roughness = 0.4
                        elif noise < -0.5:
                            biome = "mountain"
                            fertility = 0.3
                            humidity = 0.4
                            roughness = 0.7
                        else:
                            biome = "grassland"
                            fertility = 0.6
                            humidity = 0.5
                            roughness = 0.3
                    else:
                        # Polar/near-polar: tundra / mountain / frozen desert
                        if noise > 0.3:
                            biome = "mountain"
                            fertility = 0.2
                            humidity = 0.4
                            roughness = 0.8
                        elif noise < -0.4:
                            biome = "desert"
                            fertility = 0.15
                            humidity = 0.2
                            roughness = 0.3
                        else:
                            biome = "grassland"
                            fertility = 0.35
                            humidity = 0.4
                            roughness = 0.4

                    temp_offset = base_temp_offset

                region = Region(
                    col=col,
                    row=row,
                    biome=biome,
                    temp_offset=temp_offset,
                    humidity=humidity,
                    fertility=fertility,
                    roughness=roughness,
                    water=is_water,
                )
                row_regions.append(region)
            regions.append(row_regions)

        self.regions = regions

        # Normalise rows/cols and tile sizes to actual grid
        self.rows = len(self.regions)
        self.cols = len(self.regions[0]) if self.rows > 0 else 0
        if self.rows > 0 and self.cols > 0:
            self.tile_w = self.width / self.cols
            self.tile_h = self.height / self.rows

    def _initialise_food_budget(self) -> None:
        """Derive food capacity & productivity from land fertility."""

        if not self.regions:
            # Fallback: keep a modest default
            self.food_capacity = 200
            self._productivity_per_tick = self.base_food_nutrition * 0.1
            self._base_productivity_per_tick = self._productivity_per_tick
            self.food_energy_pool = self.food_capacity * self.base_food_nutrition * 0.25
            return

        land_tiles = [reg for row in self.regions for reg in row if not reg.water]
        total_fertility = sum(reg.fertility for reg in land_tiles)
        avg_fertility = total_fertility / max(1, len(land_tiles))
        self.avg_humidity = sum(reg.humidity for reg in land_tiles) / max(1, len(land_tiles))

        self.land_fraction = len(land_tiles) / max(1, self.rows * self.cols)
        self.land_area = self.width * self.height * self.land_fraction

        # Capacity grows with fertile land area; keep within reasonable bounds
        self.food_capacity = max(80, int(total_fertility * 2.5))

        # Productivity scales with average fertility; tuned to roughly match previous pacing
        self._productivity_per_tick = self.base_food_nutrition * (0.12 * avg_fertility + 0.05)
        self._base_productivity_per_tick = self._productivity_per_tick

        # Start with some biomass reserve but not a full map of food
        self.food_energy_pool = self.food_capacity * self.base_food_nutrition * 0.3

    def estimate_carrying_capacity(
        self,
        energy_need_per_tick: float,
        *,
        area_per_humlet: float = 3000.0,
    ) -> int:
        """Estimate how many humlets the landscape can sustain.

        The calculation blends available biomass, ongoing productivity, habitat
        surface area, and air quality so population limits emerge from the
        current environment instead of a hard-coded cap.
        """

        if energy_need_per_tick <= 0:
            return 0

        daily_budget = (self._productivity_per_tick * self.day_length) + self.food_energy_pool
        daily_need = energy_need_per_tick * self.day_length

        # Energy-based ceiling: how many agents could be fuelled per day.
        energy_cap = daily_budget / max(1e-6, daily_need)

        # Habitat-based ceiling: how many agents can physically coexist.
        habitat_cap = self.land_area / max(1.0, area_per_humlet)

        # Pollution/air-quality reduces the effective ceiling.
        quality_factor = 0.6 + 0.4 * self.air_quality  # 0.6–1.0

        carrying_capacity = min(energy_cap, habitat_cap) * quality_factor
        return max(10, int(carrying_capacity))

    # ------------------------------------------------------------------ #
    # Region queries
    # ------------------------------------------------------------------ #
    def get_region_at(self, x: float, y: float) -> Region | None:
        """Return the Region covering world position (x, y)."""
        if not self.regions:
            return None

        if not (0 <= x < self.width and 0 <= y < self.height):
            return None

        rows = len(self.regions)
        cols = len(self.regions[0])

        col = int(x / self.tile_w)
        row = int(y / self.tile_h)
        col = max(0, min(cols - 1, col))
        row = max(0, min(rows - 1, row))
        return self.regions[row][col]

    def get_local_temperature(self, x: float, y: float) -> float:
        """Global temperature plus this region's offset."""
        region = self.get_region_at(x, y)
        if region is None:
            return self.temperature
        return self.temperature + region.temp_offset

    def get_local_safety_factor(self, x: float, y: float) -> float:
        """Rough 'how safe is this tile?' metric in [0, 1]."""
        region = self.get_region_at(x, y)
        if region is None:
            return 0.5

        if region.biome == "forest":
            base = 0.8
        elif region.biome == "grassland":
            base = 0.7
        elif region.biome == "water":
            base = 0.4
        elif region.biome == "desert":
            base = 0.45
        elif region.biome == "mountain":
            base = 0.5
        else:
            base = 0.5

        temp = self.get_local_temperature(x, y)
        temp_penalty = max(0.0, abs(temp - 20.0) / 25.0)
        safety = base * self.air_quality * (1.0 - 0.4 * temp_penalty)
        return max(0.0, min(1.0, safety))

    # ------------------------------------------------------------------ #
    # Object management
    # ------------------------------------------------------------------ #
    def add_object(self, obj: WorldObject) -> None:
        self.objects.append(obj)
        self.object_index.insert(obj, obj.x, obj.y)

    def remove_object(self, obj: WorldObject) -> None:
        """Remove an object from the world and spatial index if present."""
        try:
            self.objects.remove(obj)
        except ValueError:
            return

        self.object_index.remove(obj, obj.x, obj.y)

    def query_objects_near(
        self,
        x: float,
        y: float,
        radius: float,
        classes: Sequence[Type[WorldObject]] | None = None,
    ) -> Iterable[WorldObject]:
        """Return objects within ``radius`` using the spatial index."""

        candidates = self.object_index.query_radius_wrapped(
            x,
            y,
            radius,
            self.width,
            self.height,
        )

        if not classes:
            return candidates

        return (obj for obj in candidates if isinstance(obj, tuple(classes)))

    def _spawn_random_food(self, count: int = 8) -> float:
        """Spawn food in clustered, biome-aware patches.

        Returns the total nutrition added so the biomass pool can be debited
        accurately when nutrition varies by biome.
        """

        energy_used = 0.0

        if not self.regions:
            # fallback: uniform scatter
            for _ in range(count):
                x = random.uniform(0, self.width)
                y = random.uniform(0, self.height)
                food = Food(x, y, nutrition=self.base_food_nutrition)
                self.add_object(food)
                energy_used += food.nutrition
            return energy_used

        def near_water(reg: Region) -> bool:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr = reg.row + dr
                    cc = reg.col + dc
                    if 0 <= rr < self.rows and 0 <= cc < self.cols:
                        if self.regions[rr][cc].water:
                            return True
            return False

        candidates: list[Region] = []
        for row in self.regions:
            for reg in row:
                if reg.water:
                    continue

                w = 1.0 + 5.0 * reg.fertility + 2.0 * reg.humidity
                if reg.biome == "forest":
                    w *= 1.4
                if near_water(reg):
                    w *= 1.6

                if w > 0:
                    candidates.extend([reg] * max(1, int(w)))

        if not candidates:
            # as a last resort, fall back to uniform
            for _ in range(count):
                x = random.uniform(0, self.width)
                y = random.uniform(0, self.height)
                food = Food(x, y, nutrition=self.base_food_nutrition)
                self.add_object(food)
                energy_used += food.nutrition
            return energy_used

        remaining = count
        while remaining > 0:
            centre = random.choice(candidates)
            cx = (centre.col + 0.5) * self.tile_w
            cy = (centre.row + 0.5) * self.tile_h

            cluster_size = random.randint(3, 7)
            for _ in range(cluster_size):
                if remaining <= 0:
                    break

                dx = random.uniform(-0.5, 0.5) * self.tile_w
                dy = random.uniform(-0.5, 0.5) * self.tile_h
                x = max(0.0, min(self.width - 1.0, cx + dx))
                y = max(0.0, min(self.height - 1.0, cy + dy))

                fertility = max(0.0, min(1.0, centre.fertility))
                humidity = max(0.0, min(1.0, centre.humidity))
                nutrition = self.base_food_nutrition * (0.6 + 0.8 * fertility + 0.2 * humidity)

                food = Food(x, y, nutrition=nutrition)
                self.add_object(food)
                energy_used += nutrition
                remaining -= 1

        return energy_used

    # ------------------------------------------------------------------ #
    # Environment update
    # ------------------------------------------------------------------ #
    def update(self) -> None:
        """Advance environment one time step: time, climate, resources."""
        self.time += 1

        # Seasonal temperature cycle (global baseline)
        phase = 2 * math.pi * (self.time / self.season_length)
        self.temperature = 20.0 + 5.0 * math.sin(phase)

        # Day/night cycle (light level)
        day_phase = 2 * math.pi * (self.time / self.day_length)
        raw_light = 0.5 + 0.5 * math.sin(day_phase - math.pi / 2)
        raw_light = max(0.0, min(1.0, raw_light))
        self.light_level = 0.15 + 0.85 * raw_light

        # Air quality jitter (bounded between 0.7 and 1.0)
        self.air_quality = max(
            0.7,
            min(1.0, self.air_quality + random.uniform(-0.005, 0.005)),
        )

        # Climate-dependent productivity: hotter/cooler seasons and humidity
        # make plants grow faster or slower.
        temp_penalty = max(0.0, abs(self.temperature - 20.0) / 15.0)
        temp_factor = max(0.5, 1.1 - temp_penalty)
        humidity_factor = 0.6 + 0.8 * self.avg_humidity
        self._productivity_per_tick = self._base_productivity_per_tick * temp_factor * humidity_factor

        # Biomass accumulation (sunlight/rain -> plants)
        capacity_energy = self.food_capacity * self.base_food_nutrition
        self.food_energy_pool = min(
            capacity_energy, self.food_energy_pool + self._productivity_per_tick
        )

        # Food respawn draws from the energy pool and stops near carrying capacity
        if self.time - self._last_food_spawn >= self.food_respawn_interval:
            food_count = sum(1 for o in self.objects if isinstance(o, Food))
            spawn_space = max(0, self.food_capacity - food_count)

            if spawn_space > 0:
                # Prefer to grow proportionally to free space and available biomass
                max_from_pool = int(self.food_energy_pool // self.base_food_nutrition)
                logistic_limit = max(1, int(spawn_space * 0.6))
                spawn_count = min(max_from_pool, logistic_limit)

                if spawn_count > 0:
                    used = self._spawn_random_food(count=spawn_count)
                    self.food_energy_pool = max(0.0, self.food_energy_pool - used)

            self._last_food_spawn = self.time
