from __future__ import annotations

import json
import math
import random
from typing import List

import numpy as np

from .environment import Environment, Food, Rock, Shelter, Tree, StoneDeposit
from .sensors.smell import Smell
from .sensors.vision import Vision


class Humlet:
    """
    Human-like artificial life agent with:
    - Energy & health (metabolism, homeostasis)
    - Genome (traits: metabolism, speed, sense range, sociability, curiosity, etc.)
    - Simple neural "brain" that uses Maslow-ish needs + environment inputs
    - Asexual reproduction with mutation
    - Lineage tracking (id, parent_id, family_id, generation)
    - Toroidal world (wrap-around edges)
    - Higher-level motivation signals:
        * Esteem (age + offspring success)
        * Curiosity drive (only active when lower needs are satisfied)
    """

    # Class-level ID counters
    _next_id = 0
    _next_family_id = 0

    # Genetic distance threshold for speciation
    SPECIES_DISTANCE_THRESHOLD = 0.2

    # Brain sizes (fixed topology)
    # Inputs:
    #   0: hunger_need
    #   1: safety_need
    #   2: social_need
    #   3: energy_norm
    #   4: health_norm
    #   5: food_dx
    #   6: food_dy
    #   7: friend_dx
    #   8: friend_dy
    #   9: shelter_dx
    #  10: shelter_dy
    #  11: esteem_level
    #  12: curiosity_drive
    N_INPUTS = 13
    N_HIDDEN = 8
    # Outputs: [move_x, move_y, eat, reproduce, rest_intensity]
    N_OUTPUTS = 5

    def __init__(
        self,
        env: Environment,
        group_id: int | None = None,
        genome: dict | None = None,
        brain: dict | None = None,
        parent_id: int | None = None,
        family_id: int | None = None,
        generation: int = 0,
        *,
        seed: int | None = None,
    ):
        # ---------------------------
        # Identity & lineage
        # ---------------------------
        self.id = Humlet._next_id
        Humlet._next_id += 1



        self.parent_id = parent_id

        # Dedicated RNGs for reproducibility and per-agent replayability
        self.seed = seed if seed is not None else random.randrange(2**32)
        self.rng = random.Random(self.seed)
        self.np_rng = np.random.default_rng(self.seed)

        # If no family_id, this individual starts a new family/species line
        if family_id is None:
            self.family_id = Humlet._next_family_id
            Humlet._next_family_id += 1
        else:
            self.family_id = family_id

        self.generation = generation
        self.offspring_count = 0  # for esteem


        # ---------------------------
        # Position (clustered by group)
        # ---------------------------
        if group_id is None:
            self.x = self.rng.uniform(0, env.width)
            self.y = self.rng.uniform(0, env.height)
        else:
            rng = random.Random(group_id)
            center_x = rng.uniform(env.width * 0.2, env.width * 0.8)
            center_y = rng.uniform(env.height * 0.2, env.height * 0.8)
            self.x = max(0, min(env.width, center_x + self.rng.gauss(0, env.width * 0.05)))
            self.y = max(0, min(env.height, center_y + self.rng.gauss(0, env.width * 0.05)))

        # Home-range anchor: where this Humlet "grew up"
        self.home_x = self.x
        self.home_y = self.y

        self.vx = 0.0
        self.vy = 0.0

        # Facing direction (radians) for cone/raycast vision
        self.direction = self.rng.uniform(0, 2 * math.pi)

        # Use sense_range so smell & "vision" are on a similar scale
        # ---------------------------
        # Sensors
        # ---------------------------
        # Ensure sense_range exists before sensors use it
        # self.sense_range = float(self.genome.get("sense_range", 60.0))

        # # Use sense_range so smell & "vision" are on a similar scale
        # self.smell = Smell(self, smell_range=self.sense_range)


        # # Directional raycast vision (debug draw supported)
        # self.vision = Vision(
        #     owner=self,
        #     fov_deg=120,
        #     num_rays=12,
        #     vision_range=int(self.sense_range)  # keep it aligned with trait
        # )


        # ---------------------------
        # Identity / social
        # ---------------------------
        self.group_id = group_id if group_id is not None else self.rng.randrange(4)
        self.neighbor_count = 0  # updated each tick in _update_needs

        # ---------------------------
        # Physiology
        # ---------------------------
        self.max_energy = 3000.0
        self.energy = self.rng.uniform(180.0, 2500.0)

        # Digestion system
        self.stomach_capacity = 80.0
        self.stomach_content = 40.0
        self.digestion_rate = 1.0
        self.absorption_efficiency = 0.85
        self.digestion_flow = 0.0
        self.waste = 0.0

        self.max_health = 200.0
        self.health = self.max_health

        self.age = 0
        self.alive = True
        self.death_info: dict | None = None

        # Needs (0–1)
        self.hunger_need = 0.0
        self.safety_need = 0.0
        self.social_need = 0.0

        # Higher-level motivation signals (0–1)
        self.esteem_level = 0.0
        self.curiosity_drive = 0.0
        # Genome (traits)
        # ---------------------------
        if genome is None:
            self.genome = {
                "metabolism_rate": self.rng.uniform(0.02, 0.06),   # baseline energy drain
                "speed_trait": self.rng.uniform(0.8, 1.5),         # movement factor
                "sense_range": self.rng.uniform(20.0, 100.0),      # vision radius
                "aggression": self.rng.uniform(0.0, 1.0),          # 0 = peaceful, 1 = aggressive (not used yet)
                "sociability": self.rng.uniform(0.0, 1.0),         # 0 = loner, 1 = highly social
                "lifespan": self.rng.randint(4000, 20000),         # ticks
                "curiosity_trait": self.rng.uniform(0.0, 1.0),     # baseline tendency to explore

                # ---- NEW: physical traits ----
                "base_mass": self.rng.uniform(5.0, 9.0),         # kg at adulthood
                "base_height": self.rng.uniform(1.4, 2.0),         # metres at adulthood
                "frame_factor": self.rng.uniform(0.8, 1.2),        # stocky vs slender
            }
        else:
            self.genome = genome
        # Aliases for convenience
        self.metabolism_rate = self.genome["metabolism_rate"]
        self.speed_trait = self.genome["speed_trait"]
        self.sense_range = self.genome["sense_range"]
        self.aggression = self.genome["aggression"]
        self.sociability = self.genome["sociability"]
        self.lifespan = self.genome["lifespan"]
        self.curiosity_trait = self.genome["curiosity_trait"]

        # ---------------------------
        # Sensors
        # ---------------------------
        # Use sense_range so smell & "vision" are on a similar scale
        self.smell = Smell(self, smell_range=self.sense_range)

        # ---- NEW physical aliases ----
        self.base_mass = self.genome["base_mass"]
        self.base_height = self.genome["base_height"]
        self.frame_factor = self.genome["frame_factor"]

        # Current physical state (will grow over time)
        self.mass = self.base_mass * 0.4       # start ~child size
        self.height = self.base_height * 0.4
        self.effective_sense_range = float(self.sense_range)
        self.movement_scalar = 1.0

        # --- Brain activation snapshots for visualization ---
        self.last_inputs = np.zeros(self.N_INPUTS, dtype=float)
        self.last_hidden = np.zeros(self.N_HIDDEN, dtype=float)
        self.last_outputs = np.zeros(self.N_OUTPUTS, dtype=float)

        # Rest state (0–1) derived from brain output
        self.rest_intensity: float = 0.0

        # ---------------------------
        # Brain (simple 1-hidden-layer NN)
        # ---------------------------
        if brain is None:
            self.brain = {
                "W1": self.np_rng.uniform(-1, 1, (self.N_HIDDEN, self.N_INPUTS)),
                "b1": np.zeros(self.N_HIDDEN),
                "W2": self.np_rng.uniform(-1, 1, (self.N_OUTPUTS, self.N_HIDDEN)),
                "b2": np.zeros(self.N_OUTPUTS),
            }
        else:
            self.brain = brain

        # Light-weight reinforcement bookkeeping so the brain is rewarded for
        # behaviours that actually help survival (e.g. eating, moving toward
        # food) instead of drifting randomly through mutation.
        self.brain_fitness: float = 0.0
        self.last_brain_reward: float = 0.0

        # ---------------------------
        # Reproduction control
        # ---------------------------
        self.repro_cooldown = 0
        self.repro_min_energy = 30.0
        self.pregnant: bool = False
        self.gestation_timer: int = 0
        self.gestation_period: int = 0
        self.embryo_genome: dict | None = None
        self.embryo_brain: dict | None = None
        self.embryo_family_id: int | None = None
        self.embryo_seed: int | None = None

        # ---------------------------
        # Inventory (for tools like rocks)
        # ---------------------------
        self.inventory: list[object] = []
        # Simple resource carrying for Phase 1
        self.carry_wood: float = 0.0
        self.carry_stone: float = 0.0
        self.carry_capacity: float = 20.0  # total units of resources

        # Thermoregulation comfort band (°C) for metabolic scaling
        self.comfort_temp_min = 18.0
        self.comfort_temp_max = 28.0

    # ------------------------------------------------------------------ #
    # Life stage & growth
    # ------------------------------------------------------------------ #
    def _life_stage(self) -> str:
        """Coarse life stage based on age vs lifespan."""
        ratio = self.age / max(1, self.lifespan)
        if ratio < 0.15:
            return "child"
        elif ratio < 0.85:
            return "adult"
        else:
            return "elder"

    def _update_physical_growth(self, *, force_to_age: bool = False) -> None:
        """
        Update mass & height as a function of age.

        Goal:
        - Reach ~adult size (base_mass/base_height * frame_factor)
        by the time age_ratio hits 0.75 (start of 'elder').
        - Children grow quickly, then plateau as adults.
        - Elders lose a little mass.
        """
        age_ratio = self.age / max(1, self.lifespan)
        age_ratio = max(0.0, min(1.0, age_ratio))

        # -----------------------------
        # Growth curve up to adulthood
        # -----------------------------
        # We treat age_ratio = 0.75 as "fully adult".
        if age_ratio < 0.75:
            # Normalised stage from birth (0) to adult (1)
            stage = age_ratio / 0.75

            # Start at 20% of adult size, ramp to 100% by stage=1
            # Using a slight curve (stage**0.7) for faster early growth.
            growth = 0.2 + 0.8 * (stage ** 0.7)
        else:
            # From 0.75 onward, the *growth* term is fully adult
            growth = 1.0

        # -----------------------------
        # Elder mass loss after adulthood
        # -----------------------------
        # Start gently reducing mass after adulthood, more towards end of life.
        if age_ratio > 0.75:
            elder_stage = (age_ratio - 0.75) / 0.25  # 0 at 0.75 -> 1 at 1.0
            elder_stage = max(0.0, min(1.0, elder_stage))
            # Lose up to ~15% mass by end of life
            elder_factor = 1.0 - 0.15 * elder_stage
        else:
            elder_factor = 1.0

        target_mass = self.base_mass * growth * elder_factor * self.frame_factor
        target_height = self.base_height * growth

        # Smooth approach to target (no teleporting)
        # Faster growth in non-elder stages, slower drifting in old age.
        if age_ratio < 0.75:
            lerp = 0.15
        else:
            lerp = 0.07

        if force_to_age:
            self.mass = target_mass
            self.height = target_height
        else:
            self.mass += (target_mass - self.mass) * lerp
            self.height += (target_height - self.height) * lerp

        # Tie physical scale into capacity
        size_factor = (self.mass / self.base_mass) ** 0.1  # still weak scaling
        self.max_energy = 100.0 * size_factor
        self.max_health = 100.0 * size_factor

        # Stomach capacity and digestion rate scale with body size/metabolism
        mass_ratio = max(0.2, self.mass / 70.0)
        self.stomach_capacity = max(10.0, 30.0 * mass_ratio * self.frame_factor)
        base_rate = 0.05 + 0.35 * self.metabolism_rate
        self.digestion_rate = max(0.2, base_rate * (0.8 + 0.6 * mass_ratio))
        self.stomach_content = min(self.stomach_content, self.stomach_capacity)

        # Clamp current state to new maxima
        self.energy = min(self.energy, self.max_energy)
        self.health = min(self.health, self.max_health)

        # Derived functional modifiers linking physique to behaviour
        avg_mass = 70.0
        avg_height = 1.7
        mass_ratio = max(0.2, self.mass / avg_mass)
        height_ratio = max(0.2, self.height / avg_height)

        # Taller frames get a stride boost; heavier bodies pay an inertia tax
        stride_boost = 0.85 + 0.25 * math.sqrt(height_ratio)
        inertia_penalty = 1.0 / (1.0 + 0.35 * max(0.0, mass_ratio - 1.0))
        self.movement_scalar = max(0.35, min(1.4, stride_boost * inertia_penalty))

        # Eye level influences awareness; update sensors to match growth
        self.effective_sense_range = self.sense_range * (0.8 + 0.3 * height_ratio)
        if hasattr(self, "smell"):
            self.smell.range = self.effective_sense_range

        # Larger bodies need more reserves before reproducing
        size_factor = (self.mass / self.base_mass) ** 0.1
        self.repro_min_energy = 40.0 * size_factor

    def set_to_reproductive_age(self) -> None:
        """Start the humlet as an adult capable of reproducing."""

        # Aim for early adulthood so the initial population can reproduce
        # immediately without spawning elders.
        self.age = max(1, int(self.lifespan * 0.3))

        # Snap physical stats to match the chosen age so energy/health and
        # movement scaling remain consistent with life stage.
        self._update_physical_growth(force_to_age=True)


    # ------------------------------------------------------------------ #
    # Toroidal world helpers
    # ------------------------------------------------------------------ #
    def _wrapped_delta(self, env: Environment, tx: float, ty: float) -> tuple[float, float]:
        """
        Smallest dx, dy on a torus (wrap-around world).
        """
        dx = tx - self.x
        dy = ty - self.y

        half_w = env.width * 0.5
        half_h = env.height * 0.5

        if dx > half_w:
            dx -= env.width
        elif dx < -half_w:
            dx += env.width

        if dy > half_h:
            dy -= env.height
        elif dy < -half_h:
            dy += env.height

        return dx, dy

    def _wrapped_delta_from(self, env: Environment, sx: float, sy: float, tx: float, ty: float) -> tuple[float, float]:
        """Toroidal delta computed from an arbitrary source point."""
        dx = tx - sx
        dy = ty - sy

        half_w = env.width * 0.5
        half_h = env.height * 0.5

        if dx > half_w:
            dx -= env.width
        elif dx < -half_w:
            dx += env.width

        if dy > half_h:
            dy -= env.height
        elif dy < -half_h:
            dy += env.height

        return dx, dy

    def _has_line_of_sight(
        self,
        env: Environment,
        tx: float,
        ty: float,
        *,
        target: object | None = None,
        target_radius: float = 0.0,
    ) -> bool:
        """Check whether a straight path to (tx, ty) is blocked by solids.

        Uses the toroidal shortest path and treats any solid world object whose
        collision circle intersects that path (and is closer than the target)
        as occluding.
        """

        dx, dy = self._wrapped_delta(env, tx, ty)
        seg_len = math.hypot(dx, dy)
        if seg_len < 1e-6:
            return True

        # Segment from (0, 0) to (dx, dy); compare obstacles in that frame
        for obj in env.objects:
            if obj is target:
                continue
            if not getattr(obj, "solid", False):
                continue

            ox, oy = self._wrapped_delta(env, obj.x, obj.y)

            # Project obstacle centre onto the path
            t = (ox * dx + oy * dy) / (seg_len * seg_len)
            if t <= 0.0 or t >= 1.0:
                continue  # behind us or past the target

            closest_dist = abs(ox * dy - oy * dx) / seg_len
            clearance = getattr(obj, "radius", 0.0) + target_radius + 1.0
            if closest_dist < clearance:
                # Also ensure the obstacle is closer than the target itself
                along_dist = t * seg_len
                if along_dist < seg_len:
                    return False

        return True

    def _collision_radius(self) -> float:
        """Body radius scales with current height, enforcing space occupancy."""
        return max(4.0, self.height * 3.0)

    def _resolve_collisions(
        self,
        env: Environment,
        humlets: list["Humlet"],
        proposed_x: float,
        proposed_y: float,
    ) -> tuple[float, float, bool]:
        """
        Push the agent out of overlapping neighbours or solid world objects.

        Returns corrected (x, y) and a flag indicating whether a collision occurred.
        """
        collided = False
        x = proposed_x
        y = proposed_y
        self_radius = self._collision_radius()

        # Resolve against other humlets
        for other in humlets:
            if other is self or not other.alive:
                continue
            dx, dy = self._wrapped_delta_from(env, x, y, other.x, other.y)
            dist = math.hypot(dx, dy)
            min_sep = self_radius + other._collision_radius()
            if dist < 1e-6 or dist >= min_sep:
                continue

            collided = True
            overlap = min_sep - dist
            nx = dx / (dist + 1e-6)
            ny = dy / (dist + 1e-6)
            x -= nx * overlap
            y -= ny * overlap
            x %= env.width
            y %= env.height

        # Resolve against solid world objects (rocks, trees, shelters, stone)
        for obj in env.objects:
            if not getattr(obj, "solid", False):
                continue
            dx, dy = self._wrapped_delta_from(env, x, y, obj.x, obj.y)
            dist = math.hypot(dx, dy)
            min_sep = self_radius + getattr(obj, "radius", 6.0)
            if dist < 1e-6 or dist >= min_sep:
                continue

            collided = True
            overlap = min_sep - dist
            nx = dx / (dist + 1e-6)
            ny = dy / (dist + 1e-6)
            x -= nx * overlap
            y -= ny * overlap
            x %= env.width
            y %= env.height

        return x, y, collided


    # ------------------------------------------------------------------ #
    # Neural network forward pass
    # ------------------------------------------------------------------ #
    def _brain_forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass of the Humlet's neural network.
        Stores activation snapshots for visualisation.
        """
        # Unpack network
        W1 = self.brain["W1"]
        b1 = self.brain["b1"]
        W2 = self.brain["W2"]
        b2 = self.brain["b2"]

        # Store inputs for brain visualisation
        self.last_inputs = np.array(inputs, copy=True)

        # -----------------------------
        # Hidden layer (tanh)
        # -----------------------------
        h_raw = W1 @ inputs + b1
        h = np.tanh(h_raw)

        # Store hidden activations
        self.last_hidden = np.array(h, copy=True)

        # -----------------------------
        # Output layer (raw signals)
        # -----------------------------
        o_raw = W2 @ h + b2

        # Movement outputs: tanh range [-1, 1]
        move_x = math.tanh(o_raw[0])
        move_y = math.tanh(o_raw[1])

        # Action outputs: numerically safe sigmoid
        def safe_sigmoid(x):
            if x >= 0:
                z = math.exp(-x)
                return 1 / (1 + z)
            else:
                z = math.exp(x)
                return z / (1 + z)

        eat = safe_sigmoid(o_raw[2])
        reproduce = safe_sigmoid(o_raw[3])
        rest = safe_sigmoid(o_raw[4])

        # Final output vector
        out = np.array([move_x, move_y, eat, reproduce, rest], dtype=float)

        # Store for brain diagram
        self.last_outputs = out

        return out

    def _brain_learn(self, reward: float) -> None:
        """Lightweight reinforcement step to nudge weights using the last pass.

        The update is deliberately tiny so the simulation remains stable while
        still letting brains adapt in-run instead of relying solely on genetic
        mutation. The reward is squashed to [-1, 1] so outliers do not explode
        the gradients.
        """

        if (
            getattr(self, "last_inputs", None) is None
            or getattr(self, "last_hidden", None) is None
            or getattr(self, "last_outputs", None) is None
        ):
            return

        reward_scale = math.tanh(float(reward))
        if abs(reward_scale) < 1e-6:
            return

        lr = 0.0025
        inputs = self.last_inputs
        hidden = self.last_hidden
        outputs = self.last_outputs

        W1 = self.brain["W1"]
        W2 = self.brain["W2"]

        # Encourage outputs that matched the reward signal
        grad_out = reward_scale * outputs
        grad_hidden = (1.0 - hidden**2) * (W2.T @ grad_out)

        self.brain["W2"] = W2 + lr * np.outer(grad_out, hidden)
        self.brain["b2"] = self.brain["b2"] + lr * grad_out
        self.brain["W1"] = W1 + lr * np.outer(grad_hidden, inputs)
        self.brain["b1"] = self.brain["b1"] + lr * grad_hidden

    # ------------------------------------------------------------------ #
    # Brain evaluation helpers (simple reinforcement signal)
    # ------------------------------------------------------------------ #
    def _brain_quality(self) -> float:
        """Convert the running fitness score into a 0–1 confidence value."""

        return 1.0 / (1.0 + math.exp(-self.brain_fitness))

    def _nearest_food_distance(self, env: Environment) -> float:
        """Return the Euclidean distance to the nearest food (wrapped)."""

        min_d2 = float("inf")
        for obj in env.objects:
            if isinstance(obj, Food):
                dx, dy = self._wrapped_delta(env, obj.x, obj.y)
                d2 = dx * dx + dy * dy
                if d2 < min_d2:
                    min_d2 = d2

        return math.sqrt(min_d2) if min_d2 < float("inf") else float("inf")

    def _update_brain_reward(
        self,
        pre: dict,
        post: dict,
        eat_signal: float,
        repro_signal: float,
    ) -> None:
        """
        Compute a shaped reward for the last tick and update a running fitness.

        The reward encourages:
        - Gaining energy/health (eating, avoiding damage).
        - Moving closer to food when hungry.
        - Avoiding collisions and hunger crises.
        """

        reward = 0.0

        # Energy / health deltas (scaled down to avoid huge jumps)
        reward += 0.02 * (post["energy"] - pre["energy"])
        reward += 0.01 * (post["health"] - pre["health"])

        # Hunger relief is a positive signal; starving is penalised heavily
        reward += 0.4 * (pre["hunger"] - post["hunger"])
        if post["hunger"] > 0.8:
            reward -= 0.5

        # Move toward food when hungry (if any food exists)
        if pre["food_dist"] < float("inf") and post["food_dist"] < float("inf"):
            distance_delta = pre["food_dist"] - post["food_dist"]
            if pre["hunger"] > 0.35:
                reward += 0.25 * distance_delta / max(1.0, pre["food_dist"])

        # Collisions or reckless reproduction are negative
        if post.get("collided", False):
            reward -= 0.2
        if repro_signal > 0.5 and post["energy"] < self.repro_min_energy:
            reward -= 0.3

        # Mild living bonus to keep alive agents from degrading to -inf
        reward += 0.01

        # Exponential moving average to smooth noisy rewards
        self.brain_fitness = 0.98 * self.brain_fitness + 0.02 * reward
        self.last_brain_reward = reward

        # Online learning step so the brain can adapt within a lifetime
        self._brain_learn(reward)

    def _should_eat(self, eat_signal: float, hunger_level: float) -> bool:
        """
        Adaptive eating decision that trusts the brain more as fitness rises.
        Reduces hard-coded thresholds so the controller is actually validated
        against energy outcomes.
        """

        trust = self._brain_quality()
        threshold = 0.55 - 0.30 * hunger_level - 0.20 * trust
        threshold = max(0.05, min(0.7, threshold))

        return eat_signal >= threshold or hunger_level > 0.6

    def _should_reproduce(self, repro_signal: float, energy_norm: float) -> bool:
        """Adaptive reproduction gate that leans on fitter brains."""

        trust = self._brain_quality()
        threshold = 0.70 - 0.25 * energy_norm - 0.15 * trust
        threshold = max(0.3, min(0.8, threshold))
        return repro_signal >= threshold

    # ------------------------------------------------------------------ #
    # Local environment sampling (biome-aware)
    # ------------------------------------------------------------------ #
    def _sample_local_env(self, env: Environment) -> dict:
        """
        Query the environment for the region at this Humlet's position.
        Returns a small dict of useful biome-related factors.
        """
        info = {
            "biome": "plain",
            "water": 0.0,
            "fertility": 0.5,
            "humidity": 0.5,
            "roughness": 0.0,
        }

        if not hasattr(env, "get_region_at"):
            return info

        region = env.get_region_at(self.x, self.y)
        if region is None:
            return info

        biome = getattr(region, "biome", "plain")
        water = 1.0 if getattr(region, "water", False) else 0.0
        fertility = float(getattr(region, "fertility", 0.5))
        humidity = float(getattr(region, "humidity", 0.5))
        roughness = float(getattr(region, "roughness", 0.0))

        info["biome"] = biome
        info["water"] = max(0.0, min(1.0, water))
        info["fertility"] = max(0.0, min(1.0, fertility))
        info["humidity"] = max(0.0, min(1.0, humidity))
        info["roughness"] = max(0.0, min(1.0, roughness))

        return info

    # ------------------------------------------------------------------ #
    # Sensing & needs
    # ------------------------------------------------------------------ #
    def _update_needs(
        self,
        env: Environment,
        humlets: List["Humlet"],
        *,
        spatial_index=None,
    ) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        """
        Update hunger_need, safety_need, social_need and return:
        - blended smell+vision direction to nearby food (dx, dy) in [-1, 1]
        - direction to nearest same-group humlet (dx, dy) in [-1, 1]
        - direction to nearest shelter (dx, dy) in [-1, 1]
        Uses toroidal distances.
        """
        # ------------------------
        # Hunger (0 = full, 1 = starving)
        # ------------------------
        energy_hunger = 1.0 - (self.energy / self.max_energy)
        energy_hunger = max(0.0, min(1.0, energy_hunger))

        if self.stomach_capacity > 0:
            stomach_satiety = self.stomach_content / self.stomach_capacity
        else:
            stomach_satiety = 0.0

        stomach_satiety = max(0.0, min(1.0, stomach_satiety))
        self.hunger_need = 0.6 * energy_hunger + 0.4 * (1.0 - stomach_satiety)
        self.hunger_need = max(0.0, min(1.0, self.hunger_need))

        # ------------------------
        # Safety: health + local environment
        # ------------------------
        local_safety = 0.5
        if hasattr(env, "get_local_safety_factor"):
            local_safety = env.get_local_safety_factor(self.x, self.y)
        else:
            # Fallback: simple global heuristic
            unsafe_env = 0.0
            if env.temperature < 10 or env.temperature > 35 or env.air_quality < 0.8:
                unsafe_env = 0.4
            local_safety = max(0.0, 1.0 - unsafe_env)

        if self._near_shelter(env):
            local_safety = min(1.0, local_safety + 0.2)

        health_factor = self.health / self.max_health

        # If either health or environment is bad, safety_need rises
        self.safety_need = 1.0 - 0.5 * (health_factor + local_safety)
        self.safety_need = max(0.0, min(1.0, self.safety_need))

        # ------------------------
        # Social: loneliness vs nearby peers
        # ------------------------
        if spatial_index is not None:
            candidates = spatial_index.query_radius_wrapped(
                self.x,
                self.y,
                self.effective_sense_range,
                env.width,
                env.height,
            )
        else:
            candidates = humlets

        neighbors = 0
        for other in candidates:
            if other is self or not getattr(other, "alive", False):
                continue
            dx, dy = self._wrapped_delta(env, other.x, other.y)
            d2 = dx * dx + dy * dy
            if d2 < (self.effective_sense_range ** 2):
                neighbors += 1

        self.neighbor_count = neighbors

        if neighbors == 0:
            self.social_need = self.sociability
        else:
            self.social_need = self.sociability * (1.0 / (neighbors + 1))
        self.social_need = max(0.0, min(1.0, self.social_need))

        # ------------------------
        # Direction helpers
        # ------------------------
        def _dir_to(target_x: float, target_y: float) -> tuple[float, float]:
            dx, dy = self._wrapped_delta(env, target_x, target_y)
            dist = math.hypot(dx, dy)
            if dist < 1e-6 or dist > self.effective_sense_range:
                return 0.0, 0.0
            # scale by sense_range so result is in roughly [-1, 1]
            return dx / self.effective_sense_range, dy / self.effective_sense_range


        # ------------------------
        # Food direction: smell + simple vision
        # ------------------------
        # Smell component
        smell_dx, smell_dy = 0.0, 0.0
        if hasattr(self, "smell") and self.smell is not None:
            smell_dx, smell_dy = self.smell.sense(env)

        # Vision component: nearest food by distance within sense range
        vision_dx, vision_dy = 0.0, 0.0
        min_food_d2 = float("inf")
        for obj in env.query_objects_near(self.x, self.y, self.effective_sense_range, classes=(Food,)):
            dx, dy = self._wrapped_delta(env, obj.x, obj.y)
            d2 = dx * dx + dy * dy
            if d2 < min_food_d2 and d2 <= (self.effective_sense_range ** 2):
                if not self._has_line_of_sight(env, obj.x, obj.y, target=obj, target_radius=getattr(obj, "radius", 0.0)):
                    continue
                min_food_d2 = d2
                vision_dx, vision_dy = _dir_to(obj.x, obj.y)

        # Blend smell + vision
        w_smell = 0.7
        w_vision = 0.3
        blended_dx = w_smell * smell_dx + w_vision * vision_dx
        blended_dy = w_smell * smell_dy + w_vision * vision_dy

        # Normalise blended vector
        mag = math.hypot(blended_dx, blended_dy)
        if mag > 1e-6:
            nearest_food_dx = blended_dx / mag
            nearest_food_dy = blended_dy / mag
        else:
            nearest_food_dx = 0.0
            nearest_food_dy = 0.0

        # ------------------------
        # Nearest shelter (still distance/vision-based)
        # ------------------------
        nearest_shelter_dx, nearest_shelter_dy = 0.0, 0.0
        min_shelter_d2 = float("inf")

        for obj in env.query_objects_near(self.x, self.y, self.effective_sense_range, classes=(Shelter,)):
            dx, dy = self._wrapped_delta(env, obj.x, obj.y)
            d2 = dx * dx + dy * dy
            if d2 < min_shelter_d2:
                min_shelter_d2 = d2
                nearest_shelter_dx, nearest_shelter_dy = _dir_to(obj.x, obj.y)

        # ------------------------
        # Nearest same-group humlet
        # ------------------------
        nearest_friend_dx, nearest_friend_dy = 0.0, 0.0
        min_friend_d2 = float("inf")
        for other in humlets:
            if other is self or not other.alive:
                continue
            if other.group_id != self.group_id:
                continue
            dx, dy = self._wrapped_delta(env, other.x, other.y)
            d2 = dx * dx + dy * dy
            if d2 < min_friend_d2:
                min_friend_d2 = d2
                nearest_friend_dx, nearest_friend_dy = _dir_to(other.x, other.y)

        return (
            (nearest_food_dx, nearest_food_dy),
            (nearest_friend_dx, nearest_friend_dy),
            (nearest_shelter_dx, nearest_shelter_dy),
        )

    def _near_shelter(self, env: Environment, radius: float = 16.0) -> bool:
        """Check if the Humlet is within `radius` of a shelter."""

        radius2 = radius * radius
        for obj in env.query_objects_near(self.x, self.y, radius, classes=(Shelter,)):
            dx, dy = self._wrapped_delta(env, obj.x, obj.y)
            if dx * dx + dy * dy <= radius2:
                return True
        return False

    # ------------------------------------------------------------------ #
    # Higher-level motivations (esteem + curiosity)
    # ------------------------------------------------------------------ #
    def _update_motivations(self):
        """
        Compute esteem_level and curiosity_drive (both 0–1).
        esteem_level ~ combination of age ratio and offspring count.
        curiosity_drive ~ curiosity_trait gated by lower needs (Maslow).
        """
        # Esteem: success = long life + offspring
        age_ratio = min(1.0, self.age / max(1, self.lifespan))
        offspring_ratio = min(1.0, self.offspring_count / 5.0)  # saturate at 5 kids

        esteem = 0.5 * age_ratio + 0.5 * offspring_ratio
        self.esteem_level = max(0.0, min(1.0, esteem))

        # Curiosity: only strong when lower needs are satisfied
        self.curiosity_drive = (
            self.curiosity_trait *
            (1.0 - self.hunger_need) *
            (1.0 - self.safety_need) *
            (1.0 - self.social_need)
        )
        self.curiosity_drive = max(0.0, min(1.0, self.curiosity_drive))

    # ------------------------------------------------------------------ #
    # Genetics: distance + mutation
    # ------------------------------------------------------------------ #
    @staticmethod
    def genetic_distance(g1: dict, g2: dict) -> float:
        """
        Numeric genetic distance between two genomes, normalised per trait.

        Raw absolute differences made "lifespan" dominate (changes of a few
        hundred ticks dwarfed 1–2% tweaks on other traits), which caused new
        families to appear almost every generation. Normalising by expected
        trait ranges makes the score roughly scale-free and keeps speciation
        tied to meaningful shifts.
        """

        # Trait scales are derived from their intended clamp ranges
        trait_ranges = {
            "metabolism_rate": 0.07,  # 0.01–0.08
            "speed_trait": 1.2,       # 0.6–1.8
            "sense_range": 280.0,     # 40–320
            "aggression": 1.0,        # 0–1
            "sociability": 1.0,       # 0–1
            "lifespan": 16000.0,      # 4_000–20_000
            "curiosity_trait": 1.0,   # 0–1
            "base_mass": 120.0,       # 30–150
            "base_height": 1.5,       # 1.0–2.5
            "frame_factor": 1.0,      # 0.5–1.5
        }

        keys = g1.keys() & g2.keys()
        if not keys:
            return 0.0

        diffs: list[float] = []
        for k in keys:
            v1 = g1[k]
            v2 = g2[k]
            if not isinstance(v1, (int, float)) or not isinstance(v2, (int, float)):
                continue

            scale = trait_ranges.get(k, max(abs(float(v1)), abs(float(v2)), 1.0))
            delta = abs(float(v1) - float(v2)) / scale
            diffs.append(min(delta, 1.0))

        if not diffs:
            return 0.0
        return sum(diffs) / len(diffs)

    def _is_new_species(self, child_genome: dict) -> bool:
        """
        Decide whether a child genome is 'far enough' from this
        individual to count as a new species / family line.
        """
        d = Humlet.genetic_distance(self.genome, child_genome)
        return d >= Humlet.SPECIES_DISTANCE_THRESHOLD

    # ------------------------------------------------------------------ #
    # Reproduction with mutation
    # ------------------------------------------------------------------ #
    def _mutate_genome(self, parent_genome: dict) -> dict:
        """
        Clone the parent's genome with *rare, small* mutations.

        - Only some genes mutate each generation (mutation_prob).
        - Most mutations are ~2% (sd), so drift is slower.
        - Per-trait clamping keeps things in plausible ranges.
        """
        mutation_prob = 0.25   # only 25% of genes mutate per generation
        small_sd = 0.02        # 2% multiplicative noise

        def clamp(name: str, value: float) -> float:
            # Per-trait safety bounds (tune as you like)
            if name == "metabolism_rate":
                return max(0.01, min(0.08, value))
            if name == "speed_trait":
                return max(0.6, min(1.8, value))
            if name == "sense_range":
                return max(40.0, min(320.0, value))
            if name == "aggression":
                return max(0.0, min(1.0, value))
            if name == "sociability":
                return max(0.0, min(1.0, value))
            if name == "curiosity_trait":
                return max(0.0, min(1.0, value))
            if name == "base_mass":
                return max(30.0, min(150.0, value))
            if name == "base_height":
                return max(1.0, min(2.5, value))
            if name == "frame_factor":
                return max(0.5, min(1.5, value))
            # default: just prevent negatives
            return max(0.0, value)

        child_genome: dict = {}

        for k, v in parent_genome.items():
            if k == "lifespan":
                # Much smaller lifespan noise, around ±200 ticks
                delta = int(self.rng.gauss(0, 200))
                child_genome[k] = max(800, v + delta)
                continue

            # Sometimes: no mutation at all → straight copy
            if self.rng.random() > mutation_prob:
                child_genome[k] = v
                continue

            # Otherwise: small multiplicative tweak
            factor = self.rng.gauss(1.0, small_sd)
            mutated = v * factor
            child_genome[k] = clamp(k, mutated)

        return child_genome

    def _mutate_brain(self, parent_brain: dict) -> dict:
        child_brain: dict = {}

        tweak_prob = 0.08
        add_prob   = 0.03
        del_prob   = 0.02
        noise_sd   = 0.15
        tiny       = 1e-3

        # stability knobs
        w_clip     = 3.0          # stop runaway weights
        add_sd     = 0.25         # gentler than 0.5

        for key in ("W1", "b1", "W2", "b2"):
            arr = np.array(parent_brain[key], copy=True)

            if key in ("b1", "b2"):
                mask = self.np_rng.random(arr.shape) < tweak_prob
                arr = arr + self.np_rng.normal(0.0, noise_sd, size=arr.shape) * mask
                child_brain[key] = np.clip(arr, -w_clip, w_clip)
                continue

            # --- IMPORTANT: decide existing/zero off the *original* array ---
            was_existing = np.abs(arr) > tiny
            was_zero     = ~was_existing

            # 1) Tweak existing connections only (optional, but usually cleaner)
            tweak_mask = (self.np_rng.random(arr.shape) < tweak_prob) & was_existing
            arr = arr + self.np_rng.normal(0.0, noise_sd, size=arr.shape) * tweak_mask

            # 2) Delete some existing connections
            del_mask = (self.np_rng.random(arr.shape) < del_prob) & was_existing
            arr[del_mask] = 0.0

            # 3) Add some new connections only where it was *truly* zero
            add_mask = (self.np_rng.random(arr.shape) < add_prob) & was_zero
            arr[add_mask] = self.np_rng.normal(0.0, add_sd, size=arr.shape)[add_mask]

            child_brain[key] = np.clip(arr, -w_clip, w_clip)

        return child_brain

    def _gestation_ticks(self) -> int:
        """Derive a gestation length tied to lifespan to slow explosive cloning."""
        base = int(self.lifespan * 0.08)
        return max(400, min(1000, base))

    def _give_birth(self, env: Environment, newborns: list["Humlet"]):
        """Finish gestation and spawn the stored embryo."""
        if not self.pregnant or self.embryo_genome is None or self.embryo_brain is None:
            return

        child = Humlet(
            env,
            group_id=self.group_id,
            genome=self.embryo_genome,
            brain=self.embryo_brain,
            parent_id=self.id,
            family_id=self.embryo_family_id,
            generation=self.generation + 1,
            seed=self.embryo_seed,
        )

        child.x = (self.x + self.rng.uniform(-5, 5)) % env.width
        child.y = (self.y + self.rng.uniform(-5, 5)) % env.height

        newborns.append(child)
        self.offspring_count += 1

        # Post-birth fatigue: force a cooldown and some energy loss to slow churn
        self.energy = max(0.0, self.energy - self.max_energy * 0.1)
        self.health = max(0.0, self.health - 3.0)
        self.repro_cooldown = max(self.repro_cooldown, 300)

        # Reset gestation state
        self.pregnant = False
        self.gestation_timer = 0
        self.gestation_period = 0
        self.embryo_genome = None
        self.embryo_brain = None
        self.embryo_family_id = None
        self.embryo_seed = None

    def estimated_energy_need(self, env: Environment) -> float:
        """Rough per-tick energy demand for carrying capacity estimates."""

        mass = max(1.0, self.mass)
        mass_ratio = mass / 70.0
        metabolic_intensity = 0.7 + 10.0 * max(0.0, self.metabolism_rate - 0.02)
        basal_burn = 0.12 * (mass_ratio ** 0.75) * metabolic_intensity

        avg_speed = 0.6 * self.speed_trait * self.movement_scalar
        locomotion_cost = 0.04 * mass_ratio * (avg_speed ** 2)
        locomotion_cost *= (0.9 + 0.15 * self.speed_trait)
        locomotion_cost *= (0.8 + 0.4 * getattr(env, "air_density", 1.0))

        if hasattr(env, "get_local_temperature"):
            local_temp = env.get_local_temperature(self.x, self.y)
        else:
            local_temp = getattr(env, "temperature", 20.0)

        if local_temp < self.comfort_temp_min:
            temp_delta = self.comfort_temp_min - local_temp
            thermo_cost = 0.004 * mass_ratio * temp_delta
        elif local_temp > self.comfort_temp_max:
            temp_delta = local_temp - self.comfort_temp_max
            thermo_cost = 0.003 * mass_ratio * temp_delta
        else:
            thermo_cost = 0.0

        return max(0.05, basal_burn + locomotion_cost + thermo_cost)

    def maybe_reproduce(
        self,
        env: Environment,
        newborns: list["Humlet"],
        population_size: int,
        max_population: int
    ):
        """Asexual reproduction: clone with mutation if energy & cooldown allow."""
        if self.pregnant:
            return
        if not self.alive:
            return
        if population_size >= max_population:
            return
        if self.repro_cooldown > 0:
            self.repro_cooldown -= 1
            return
        if self.energy < self.repro_min_energy:
            return
        # Poorly performing brains are less likely to successfully clone
        if self.brain_fitness < -2.0:
            return

        # Only adults can reproduce
        if self._life_stage() != "adult":
            return

        # Local environment gates reproduction: harsh climates or barren land slow it down
        local_env = self._sample_local_env(env)
        local_temp = env.get_local_temperature(self.x, self.y) if hasattr(env, "get_local_temperature") else env.temperature

        fertility_factor = 0.5 + 0.8 * local_env.get("fertility", 0.5)
        humidity_penalty = 0.2 if local_env.get("humidity", 0.5) < 0.25 else 0.0
        temp_penalty = max(0.0, abs(local_temp - 22.0) / 20.0)

        # Chance to reproduce increases with surplus energy, but is damped by climate
        surplus = (self.energy - self.repro_min_energy) / (self.max_energy - self.repro_min_energy + 1e-6)
        surplus = max(0.0, min(1.0, surplus))
        surplus *= fertility_factor
        surplus *= max(0.0, 1.0 - (temp_penalty + humidity_penalty))
        surplus *= 0.5 + 0.5 * self._brain_quality()
        if self.rng.random() > surplus:
            return

        # Pay reproduction cost up front and lock parent into gestation
        gestation_period = self._gestation_ticks()
        upfront_cost = self.energy * 0.25  # heavier investment than before
        self.energy -= upfront_cost
        self.repro_cooldown = gestation_period + 400  # postpartum rest baked in

        # Create embryo with mutated genome & brain and stash until birth
        self.embryo_genome = self._mutate_genome(self.genome)
        self.embryo_brain = self._mutate_brain(self.brain)
        self.embryo_seed = self.rng.randrange(2**32)

        is_new_species = self._is_new_species(self.embryo_genome)
        self.embryo_family_id = None if is_new_species else self.family_id

        self.pregnant = True
        self.gestation_period = gestation_period
        self.gestation_timer = gestation_period

    def _progress_gestation(self, env: Environment, newborns: list["Humlet"]):
        if not self.pregnant:
            return

        # Gradually burn extra energy as the fetus grows
        progress = 1.0 - (self.gestation_timer / max(1, self.gestation_period))
        gestation_burn = 0.05 + 0.08 * progress
        self.energy -= gestation_burn

        # Late-stage pregnancy slows the parent and risks health if starving
        if self.energy <= 0:
            self.health = max(0.0, self.health - 0.05)

        self.gestation_timer -= 1
        if self.gestation_timer <= 0:
            self._give_birth(env, newborns)

    # ------------------------------------------------------------------ #
    # Main update per tick
    # ------------------------------------------------------------------ #
    def update(
        self,
        env: Environment,
        humlets: List["Humlet"],
        newborns: list["Humlet"],
        max_population: int,
        *,
        spatial_index=None,
    ):
        if not self.alive:
            return

        self.age += 1

        # Pregnancy advances independent of brain signals
        self._progress_gestation(env, newborns)

        # Physical growth & derived capacities
        self._update_physical_growth()

        # 1. Update needs & perception
        (food_dx, food_dy), (friend_dx, friend_dy), (shelter_dx, shelter_dy) = self._update_needs(
            env, humlets, spatial_index=spatial_index
        )

        pre_state = {
            "energy": self.energy,
            "health": self.health,
            "hunger": self.hunger_need,
            "food_dist": self._nearest_food_distance(env),
        }

        # 2. Update higher-level motivations (esteem + curiosity)
        self._update_motivations()

        # 3. Build brain input vector (N_INPUTS)
        energy_norm = self.energy / self.max_energy
        health_norm = self.health / self.max_health

        inputs = np.array([
            self.hunger_need,        # 0
            self.safety_need,        # 1
            self.social_need,        # 2
            energy_norm,             # 3
            health_norm,             # 4
            food_dx,                 # 5
            food_dy,                 # 6
            friend_dx,               # 7
            friend_dy,               # 8
            shelter_dx,              # 9
            shelter_dy,              # 10
            self.esteem_level,       # 11
            self.curiosity_drive,    # 12
        ], dtype=float)

        # 4. Brain decides actions
        move_x, move_y, eat_signal, repro_signal, rest_signal = self._brain_forward(inputs)

        # Persist rest state for downstream metabolism/regen effects
        self.rest_intensity = rest_signal

        # 4a. Curiosity-driven exploration noise
        if self.curiosity_drive > 0.0:
            angle = self.rng.uniform(0, 2 * math.pi)
            rx = math.cos(angle)
            ry = math.sin(angle)
            curiosity_strength = 0.3
            alpha = curiosity_strength * self.curiosity_drive
            move_x = (1.0 - alpha) * move_x + alpha * rx
            move_y = (1.0 - alpha) * move_y + alpha * ry

        # 4b. Rest dampens voluntary movement ("sleep" behaviour)
        if rest_signal > 0.0:
            rest_damper = 1.0 - 0.75 * rest_signal
            move_x *= rest_damper
            move_y *= rest_damper

        # 4c. Home-range pull: don't wander *too* far from birthplace
        home_dx, home_dy = self._wrapped_delta(env, self.home_x, self.home_y)
        home_dist = math.hypot(home_dx, home_dy)
        if home_dist > self.effective_sense_range * 0.8:
            # Only pull back if we're quite far from home
            pull = 0.1  # 0 = none, 1 = snap to home
            nx = home_dx / (home_dist + 1e-6)
            ny = home_dy / (home_dist + 1e-6)
            move_x = (1.0 - pull) * move_x + pull * nx
            move_y = (1.0 - pull) * move_y + pull * ny

        # Normalize movement to avoid excessive speed
        mag = math.hypot(move_x, move_y)
        if mag > 1.0:
            move_x /= mag
            move_y /= mag


        # 5. Interpret brain outputs

        # movement
        speed = 1.75 * self.speed_trait * self.movement_scalar
        local_env = self._sample_local_env(env)
        speed *= 1.0 - 0.35 * local_env.get("roughness", 0.0)
        if local_env.get("water", 0.0) > 0.5:
            speed *= 0.85
        if self.pregnant:
            preg_progress = 1.0 - (self.gestation_timer / max(1, self.gestation_period))
            speed *= max(0.45, 0.9 - 0.3 * preg_progress)
        if rest_signal > 0.0:
            speed *= 0.35 + 0.65 * (1.0 - rest_signal)
        self.vx = move_x * speed
        self.vy = move_y * speed

        # Update facing direction from movement (only if actually moving)
        if (self.vx * self.vx + self.vy * self.vy) > 1e-8:
            self.direction = math.atan2(self.vy, self.vx)


        # position update (toroidal) with collision resolution
        proposed_x = (self.x + self.vx) % env.width
        proposed_y = (self.y + self.vy) % env.height
        corrected_x, corrected_y, collided = self._resolve_collisions(
            env, humlets, proposed_x, proposed_y
        )
        self.x = corrected_x
        self.y = corrected_y
        if collided:
            self.vx *= 0.25
            self.vy *= 0.25
            impact_speed = math.hypot(self.vx, self.vy)
            impact_energy = (self.mass / 70.0) * (impact_speed ** 2)
            self.health = max(0.0, self.health - 0.02 * impact_energy)

        # eating: much easier threshold so they actually refuel
        # hunger_need = 1 - energy/max_energy, so 0.3 ~= "below ~70% energy"
        if self._should_eat(eat_signal, self.hunger_need):
            self._try_eat(env)

        # reproduction
        if self._should_reproduce(repro_signal, energy_norm):
            alive_count = sum(1 for h in humlets if h.alive)
            self.maybe_reproduce(env, newborns, population_size=alive_count, max_population=max_population)

        # Resource behaviours (Phase 1)
        self._gather_resources(env)
        self._deposit_resources(env)

        # 5. Digestion turns stored food into usable energy
        self._digest()

        # 6. Metabolic cost & environmental damage (and social + esteem effects)
        self._apply_metabolism_and_damage(env)

        # Reinforcement-style reward for the brain based on outcomes of this tick
        post_state = {
            "energy": self.energy,
            "health": self.health,
            "hunger": max(0.0, min(1.0, 1.0 - (self.energy / self.max_energy))),
            "food_dist": self._nearest_food_distance(env),
            "collided": collided,
        }
        self._update_brain_reward(pre_state, post_state, eat_signal=eat_signal, repro_signal=repro_signal)

        # 7. Death checks
        eps = 0.05  # same threshold used when clamping
        cause = None
        if self.age > self.lifespan:
            cause = "old_age"
        elif self.energy <= eps:
            cause = "starvation"
        elif self.health <= eps:
            cause = self._determine_death_cause(env)

        if cause is not None:
            self._mark_dead(cause, env)

    # ------------------------------------------------------------------ #
    # Resource gathering & depositing (Phase 1)
    # ------------------------------------------------------------------ #
    def _gather_resources(self, env: Environment):
        """
        Very simple gathering rule:
        - If near a Tree and have capacity, convert some wood into carried_wood.
        - If near a StoneDeposit and have capacity, convert some stone.
        """
        if not self.alive:
            return

        remaining_capacity = self.carry_capacity - (self.carry_wood + self.carry_stone)
        if remaining_capacity <= 0:
            return

        gather_radius2 = 8.0 ** 2
        gather_rate = 0.5  # units per tick

        for obj in list(env.query_objects_near(self.x, self.y, gather_radius2 ** 0.5, classes=(Tree, StoneDeposit))):
            dx, dy = self._wrapped_delta(env, obj.x, obj.y)
            if dx * dx + dy * dy > gather_radius2:
                continue

            if isinstance(obj, Tree) and obj.wood_amount > 0:
                amount = min(gather_rate, obj.wood_amount, remaining_capacity)
                obj.wood_amount -= amount
                self.carry_wood += amount
                remaining_capacity -= amount
                if obj.wood_amount <= 0:
                    env.remove_object(obj)
                if remaining_capacity <= 0:
                    break

            elif isinstance(obj, StoneDeposit) and obj.stone_amount > 0:
                amount = min(gather_rate, obj.stone_amount, remaining_capacity)
                obj.stone_amount -= amount
                self.carry_stone += amount
                remaining_capacity -= amount
                if obj.stone_amount <= 0:
                    env.remove_object(obj)
                if remaining_capacity <= 0:
                    break

    def _deposit_resources(self, env: Environment):
        """
        If close to the village centre, dump carried resources into village storage.
        """
        if not self.alive:
            return

        if self.carry_wood <= 0 and self.carry_stone <= 0:
            return

        # If you haven't created env.village yet, guard this or add it later
        if not hasattr(env, "village"):
            return

        vx, vy = env.village.x, env.village.y
        dx, dy = self._wrapped_delta(env, vx, vy)
        dist2 = dx * dx + dy * dy
        deposit_radius2 = 12.0 ** 2

        if dist2 <= deposit_radius2:
            if self.carry_wood > 0:
                env.village.add_resource("wood", self.carry_wood)
                self.carry_wood = 0.0
            if self.carry_stone > 0:
                env.village.add_resource("stone", self.carry_stone)
                self.carry_stone = 0.0

    # ------------------------------------------------------------------ #
    # Helpers: eating & metabolism
    # ------------------------------------------------------------------ #
    def _try_eat(self, env: Environment):
        """Consume food if within small radius (with wrap-around)."""
        eat_radius2 = 5.0 ** 2
        for obj in list(env.query_objects_near(self.x, self.y, eat_radius2 ** 0.5, classes=(Food,))):
            dx, dy = self._wrapped_delta(env, obj.x, obj.y)
            if dx * dx + dy * dy <= eat_radius2:
                space = max(0.0, self.stomach_capacity - self.stomach_content)
                if space <= 1e-3:
                    continue

                bite_cap = max(2.0, 0.15 * self.stomach_capacity)
                bite = min(space, obj.nutrition, bite_cap)
                if bite <= 0:
                    continue

                self.stomach_content += bite
                obj.nutrition -= bite
                if obj.nutrition <= 0.1:
                    env.remove_object(obj)
                break

    def _digest(self):
        """Convert stomach contents into usable energy over time."""

        self.digestion_flow = 0.0
        if self.stomach_content <= 0.0:
            return

        digest_amount = min(self.stomach_content, self.digestion_rate)
        energy_gain = digest_amount * self.absorption_efficiency

        self.stomach_content -= digest_amount
        self.energy = min(self.max_energy, self.energy + energy_gain)

        self.digestion_flow = energy_gain
        self.waste += max(0.0, digest_amount - energy_gain)

    def _apply_metabolism_and_damage(self, env: Environment):
        """Apply energy drain, social energy adjust, esteem tweak, and environmental health impacts."""
        # ---------------- Basal + movement energy cost ----------------
        rest = max(0.0, min(1.0, float(getattr(self, "rest_intensity", 0.0))))

        mass = max(1.0, self.mass)
        mass_ratio = mass / 70.0  # relative to ~average adult

        # Kleiber-like basal metabolic scaling that ties drain to body size
        metabolic_intensity = 0.7 + 10.0 * max(0.0, self.metabolism_rate - 0.02)
        basal_burn = 0.12 * (mass_ratio ** 0.75) * metabolic_intensity
        basal_burn *= 1.0 - 0.45 * rest  # resting slows basal drain

        # Movement cost scales with kinetic effort, speed trait, and air density
        speed_mag = math.hypot(self.vx, self.vy)
        locomotion_cost = 0.04 * mass_ratio * (speed_mag ** 2)
        locomotion_cost *= (0.9 + 0.15 * self.speed_trait)
        locomotion_cost *= (0.8 + 0.4 * getattr(env, "air_density", 1.0))
        locomotion_cost *= 1.0 - 0.25 * rest

        # Thermoregulation: energy cost grows with deviation from comfort band
        if hasattr(env, "get_local_temperature"):
            local_temp = env.get_local_temperature(self.x, self.y)
        else:
            local_temp = getattr(env, "temperature", 20.0)

        sheltered = self._near_shelter(env)

        if local_temp < self.comfort_temp_min:
            temp_delta = self.comfort_temp_min - local_temp
            thermo_cost = 0.008 * mass_ratio * temp_delta
        elif local_temp > self.comfort_temp_max:
            temp_delta = local_temp - self.comfort_temp_max
            thermo_cost = 0.006 * mass_ratio * temp_delta
        else:
            thermo_cost = 0.0

        if sheltered:
            thermo_cost *= 0.5

        # ---------------- Local biome effects (for energy budgeting) ----------------
        local = self._sample_local_env(env)
        biome = local["biome"]
        fertility = local["fertility"]
        humidity = local["humidity"]
        water_level = local["water"]
        roughness = local.get("roughness", 0.0)

        # Rough terrain makes moving more expensive
        locomotion_cost *= 1.0 + 0.7 * roughness

        self.energy -= (basal_burn + locomotion_cost + thermo_cost)

        # If out of energy, burn health instead (energy deficit)
        if self.energy < 0:
            self.health += self.energy  # subtract deficit from health
            self.energy = 0.0

        # ---------------- Harsh global env damage ----------------
        if env.temperature < 5 or env.temperature > 40 or env.air_quality < 0.75:
            self.health -= 0.02

        # Desert: extra dehydration / energy burn, small health hit
        if biome == "desert":
            self.energy -= 0.01
            if humidity < 0.3:
                self.health -= 0.01

        # Forest / fertile areas: slightly easier to maintain energy
        if biome in ("forest", "grassland"):
            # only give a boost if they’re at least a bit hungry
            hunger_factor = self.hunger_need
            self.energy += 0.005 * (fertility - 0.5) * hunger_factor

        # Water: slightly more energy cost to move, but protective
        if water_level > 0.5:
            self.energy -= 0.005
            if self.safety_need < 0.5:
                self.health += 0.005

        # Heat or cold stress: harsher health penalties if far outside comfort
        hot_excess = max(0.0, local_temp - self.comfort_temp_max - 2.0)
        cold_excess = max(0.0, self.comfort_temp_min - local_temp - 2.0)
        if hot_excess > 0.0:
            heat_hit = 0.004 * (1.0 + 0.5 * humidity) * hot_excess
            if sheltered:
                heat_hit *= 0.4
            self.health -= heat_hit
        elif cold_excess > 0.0:
            cold_hit = 0.003 * (1.0 + roughness) * cold_excess
            if sheltered:
                cold_hit *= 0.4
            self.health -= cold_hit

        # ---------------- Social energy adjustments ----------------
        if self.sociability > 0.05:
            if self.neighbor_count == 0:
                self.energy -= 0.01 * self.sociability
            elif self.neighbor_count >= 2:
                self.energy += 0.01 * self.sociability

        # ---------------- Esteem-based tweak ----------------
        esteem_delta = (self.esteem_level - 0.5) * 0.04  # ~[-0.02, +0.02]
        self.energy += esteem_delta

        # ---------------- Health regeneration ----------------
        # Recompute hunger from *current* energy for regen logic
        current_hunger = 1.0 - (self.energy / max(1e-6, self.max_energy))
        current_hunger = max(0.0, min(1.0, current_hunger))

        good_energy = (self.energy / max(1e-6, self.max_energy)) > 0.5
        low_hunger = current_hunger < 0.5
        safe_enough = self.safety_need < 0.6
        not_very_old = self.age < 0.9 * self.lifespan

        if good_energy and low_hunger and safe_enough and not_very_old:
            # Base regen per tick
            regen = 0.02

            # More regen if basically resting (very low speed)
            speed2 = self.vx * self.vx + self.vy * self.vy
            if speed2 < 0.05:
                regen += 0.02

            if rest > 0.2 and speed2 < 0.1:
                regen += 0.01 * rest

            # Scale with how topped-up their energy is
            energy_frac = self.energy / max(1e-6, self.max_energy)
            regen *= 0.5 + 0.5 * energy_frac  # 0.5x–1x

            self.health += regen

        # ---------------- Final clamp / cleanup ----------------
        self.energy = max(0.0, min(self.energy, self.max_energy))
        self.health = max(0.0, min(self.health, self.max_health))

        # Treat very small values as zero so they don't linger
        if self.energy < 0.05:
            self.energy = 0.0

        # Health: only snap *tiny* numerical noise to zero,
        # don't wipe out legitimate small healing
        if self.health < 1e-4:
            self.health = 0.0

    def _determine_death_cause(self, env: Environment | None) -> str:
        if env is None:
            return "other"

        local_temp = env.get_local_temperature(self.x, self.y)
        temp_extreme = (
            local_temp > self.comfort_temp_max + 5 or local_temp < self.comfort_temp_min - 5
        )
        bad_air = getattr(env, "air_quality", 1.0) < 0.75

        if temp_extreme or bad_air:
            return "exposure"
        return "injury"

    def _mark_dead(self, cause: str, env: Environment | None) -> None:
        if not self.alive and self.death_info is not None:
            return

        region = env.get_region_at(self.x, self.y) if env is not None else None

        action = "Move"
        if self.last_outputs[2] > 0.5:
            action = "Eat"
        elif self.last_outputs[3] > 0.5:
            action = "Reproduce"

        self.death_info = {
            "cause": cause,
            "region_col": getattr(region, "col", None),
            "region_row": getattr(region, "row", None),
            "hunger_need": self.hunger_need,
            "safety_need": self.safety_need,
            "social_need": self.social_need,
            "last_action": action,
            "brain_outputs": json.dumps(self.last_outputs.tolist()),
        }

        self.alive = False

    # ------------------------------------------------------------------ #
    # Visual helpers
    # ------------------------------------------------------------------ #
    def render_radius(self) -> int:
        """
        Radius in pixels for drawing this Humlet, based on mass.

        Uses a cube-root scale (like volume ~ mass) so really big Humlets
        don't explode in size. Also clamps to a sensible min/max.
        """
        # Pick a reference adult mass so "normal" Humlets look reasonable
        ref_mass = 70.0  # kg

        # Avoid divide-by-zero and silly values
        mass = max(1.0, float(self.mass))

        # Cube-root scaling: size ∝ (mass / ref_mass)^(1/3)
        scale = (mass / ref_mass) ** (1.0 / 3.0)

        # Base radius and how much it can grow
        base_radius = 1.0       # child-sized
        max_extra   = 3.0      # extra radius for very big Humlets

        radius = base_radius + max_extra * scale

        # Clamp to something sensible on screen
        radius = max(3.0, min(18.0, radius))

        return int(radius)


    # ------------------------------------------------------------------ #
    # Convenience helpers for inspector
    # ------------------------------------------------------------------ #
    def brain_param_count(self) -> int:
        return int(
            self.brain["W1"].size +
            self.brain["b1"].size +
            self.brain["W2"].size +
            self.brain["b2"].size
        )
