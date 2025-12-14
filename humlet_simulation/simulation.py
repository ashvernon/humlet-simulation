from __future__ import annotations

import json
import random
import pygame
import numpy as np

from .environment import Environment, Food, Rock, Shelter, Tree, StoneDeposit
from .humlet import Humlet
from .spatial_hash import SpatialHash
from .stats import EvolutionStats, RegionTraitStats
from .sensors.vision import Vision

class Simulation:
    """Main class to set up and run the Humlet life simulation."""

    def __init__(
        self,
        world_width: int = 800,
        world_height: int = 600,
        num_humlets: int = 200,
        panel_width: int = 260,
        margin: int = 10,
        *,
        seed: int | None = None,
        trajectory_log_path: str | None = None,
    ):
        # --- Core geometry / layout ---
        self.world_width = world_width
        self.world_height = world_height
        self.panel_width = panel_width
        self.margin = margin

        # Two inspector panels on the right
        self.num_panels = 2

        # Randomness & reproducibility
        self.base_seed = seed if seed is not None else random.randrange(2**32)
        random.seed(self.base_seed)
        np.random.seed(self.base_seed)
        self.seed_rng = random.Random(self.base_seed)
        self.agent_seeds: dict[int, int] = {}
        self.trajectory_log_path = trajectory_log_path

        # World (simulation space)
        self.env = Environment(self.world_width, self.world_height)

        # Population & limits
        self.humlets: list[Humlet] = []
        self.max_population = num_humlets

        # Per-region trait stats for heatmaps
        self.region_stats = RegionTraitStats(self.env)
        self.heatmap_mode = None  # or "met", "spd", "sns", "soc", "agg"
        self.region_stats_update_interval = 5
        self._last_region_stats_tick = -1

        # Spatial index for fast neighbour lookups (rebuilt each tick)
        self.humlet_index = SpatialHash(self.world_width, self.world_height, cell_size=80.0)

        # Initial population
        group_count = 10
        for i in range(num_humlets):
            group_id = i % group_count
            seed_value = self.seed_rng.randrange(2**32)
            humlet = Humlet(self.env, group_id=group_id, seed=seed_value)
            humlet.set_to_reproductive_age()
            self.humlets.append(humlet)
            self.agent_seeds[humlet.id] = seed_value

        # Initial resources
        for _ in range(80):
            x = self.seed_rng.uniform(0, self.world_width)
            y = self.seed_rng.uniform(0, self.world_height)
            self.env.add_object(Food(x, y, nutrition=40.0))

        for _ in range(30):
            x = self.seed_rng.uniform(0, self.world_width)
            y = self.seed_rng.uniform(0, self.world_height)
            self.env.add_object(Rock(x, y))

        for _ in range(6):
            x = self.seed_rng.uniform(0, self.world_width)
            y = self.seed_rng.uniform(0, self.world_height)
            self.env.add_object(Shelter(x, y))

        # Build initial spatial index for humlets
        self._rebuild_humlet_index()

        # Derive carrying capacity from the actual environment and agents
        self._update_population_capacity()


        # Pygame setup
        pygame.init()

        total_width = (
            self.world_width                       # world area
            + self.margin                          # gap
            + self.num_panels * self.panel_width   # two inspector panels
        )
        window_height = self.world_height

        self.screen = pygame.display.set_mode((total_width, window_height))
        pygame.display.set_caption("Humelet Simulation")
        self.clock = pygame.time.Clock()

        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 24)

        # Pause & speed
        self.paused = False
        self.fast_mode = False
        self.show_framerate = False

        # Selected humlet for inspection
        self.selected_humlet: Humlet | None = None

        # Simulation time / stats
        self.tick = 0
        self.stats = EvolutionStats()

        # Big brain overlay toggle
        self.show_brain_overlay = False

    def _append_trajectory_log(self) -> None:
        """Persist a machine-readable record of this tick's state."""

        snapshot = self.stats.latest_as_dict()
        if snapshot is None:
            return

        record = {
            "tick": self.tick,
            "base_seed": self.base_seed,
            "agent_seeds": self.agent_seeds,
            "snapshot": snapshot,
        }

        with open(self.trajectory_log_path, "a", encoding="utf-8") as f:
            json.dump(record, f)
            f.write("\n")

    def seed_manifest(self) -> dict:
        """Expose the seeds used for the run for offline replay."""

        return {
            "base_seed": self.base_seed,
            "agent_seeds": dict(self.agent_seeds),
        }

    def _estimate_energy_need_per_tick(self) -> float:
        """Average per-tick energy requirement of the current population."""

        alive = [h for h in self.humlets if h.alive]
        if not alive:
            return 0.18  # reasonable default based on basal burn

        needs = [h.estimated_energy_need(self.env) for h in alive]
        return max(0.05, sum(needs) / len(needs))

    def _update_population_capacity(self) -> None:
        """Derive carrying capacity from energy flux and habitat size."""

        energy_need = self._estimate_energy_need_per_tick()
        env_capacity = self.env.estimate_carrying_capacity(energy_need)

        # Never drop below the current alive population so deaths, not caps,
        # control decline.
        alive_count = sum(1 for h in self.humlets if h.alive)
        self.max_population = max(alive_count, env_capacity)


    # --------------------------------------------------------------- #
    # Phenotype-based colouring (nature-inspired)
    # --------------------------------------------------------------- #
    def _phenotype_color(self, h: Humlet) -> tuple[int, int, int]:
        """Compute an RGB colour for a humlet based on:
            • Aggression     → Red channel
            • Sociability    → Green channel
            • Sense range    → Blue channel
            • Health/Energy  → Brightness factor
            • Age            → Subtle dulling
        Scaled relative to the *current population* for realism.
        """

        alive = [x for x in self.humlets if x.alive]
        if len(alive) < 2:
            return (200, 200, 200)

        # ----- Population-relative min/max -----
        def get_minmax(fn):
            vals = [fn(x) for x in alive]
            return min(vals), max(vals)

        agg_min, agg_max = get_minmax(lambda x: x.aggression)
        soc_min, soc_max = get_minmax(lambda x: x.sociability)
        sen_min, sen_max = get_minmax(lambda x: x.sense_range)

        # Avoid division by zero
        eps = 1e-6

        # ----- Normalize traits -----
        agg_norm = (h.aggression - agg_min) / (agg_max - agg_min + eps)
        soc_norm = (h.sociability - soc_min) / (soc_max - soc_min + eps)
        sen_norm = (h.sense_range - sen_min) / (sen_max - sen_min + eps)

        agg_norm = max(0, min(1, agg_norm))
        soc_norm = max(0, min(1, soc_norm))
        sen_norm = max(0, min(1, sen_norm))

        # ----- Base RGB (biological meaning) -----
        base_R = 50 + int(205 * agg_norm)     # Aposematic / dominance
        base_G = 50 + int(205 * soc_norm)     # Social affiliation
        base_B = 40 + int(215 * sen_norm)     # Perceptual ability

        # ----- Condition (health + energy) -----
        cond = 0.5 * (h.health / h.max_health) + \
               0.5 * (h.energy / h.max_energy)

        cond = max(0.0, min(1.0, cond))

        # ----- Age dulling -----
        age_ratio = min(1.0, h.age / max(1, h.lifespan))
        age_factor = 0.8 + 0.2 * (1.0 - age_ratio)  # young=1.0, old~0.8
        cond *= age_factor

        # Nonlinear brightness to emphasise low health
        brightness = cond ** 0.7

        # ----- Apply brightness -----
        R = int(base_R * brightness)
        G = int(base_G * brightness)
        B = int(base_B * brightness)

        return (max(0, min(255, R)),
                max(0, min(255, G)),
                max(0, min(255, B)))

    # ------------------------------------------------------------------ #
    # Biome background rendering
    # ------------------------------------------------------------------ #
    def _draw_biome_background(self) -> None:
        """Draw a coarse biome map (water / grassland / forest / desert / mountain)
        as the background, tinted by the current day/night light level.
        """
        env = self.env

        # If environment hasn't been upgraded yet, bail out safely
        if not hasattr(env, "regions"):
            return

        tile_w = env.tile_w
        tile_h = env.tile_h
        light = getattr(env, "light_level", 1.0)  # 0..1

        # Simple brightness scaling so nights are darker
        brightness = 0.3 + 0.7 * light  # 0.3 at night, 1.0 at full day
        brightness = max(0.1, min(1.0, brightness))

        for row in range(env.rows):
            for col in range(env.cols):
                region = env.regions[row][col]
                biome = region.biome

                # Base colours per biome (before lighting)
                if biome == "water":
                    base = (20, 60, 130)
                elif biome == "grassland":
                    base = (40, 120, 40)
                elif biome == "forest":
                    base = (20, 80, 30)
                elif biome == "desert":
                    base = (170, 150, 60)
                elif biome == "mountain":
                    base = (90, 90, 105)
                else:
                    base = (50, 50, 50)

                # Subtle tweak using fertility/humidity
                fert = getattr(region, "fertility", 0.5)  # 0..1
                hum = getattr(region, "humidity", 0.5)    # 0..1

                r, g, b = base
                # Slightly boost green with fertility on land
                if not getattr(region, "water", False):
                    g = int(g + 30 * fert)
                else:
                    # Water gets a little extra blue with humidity
                    b = int(b + 20 * hum)

                # Apply day/night brightness
                r = int(r * brightness)
                g = int(g * brightness)
                b = int(b * brightness)

                color = (
                    max(0, min(255, r)),
                    max(0, min(255, g)),
                    max(0, min(255, b)),
                )

                x = int(col * tile_w)
                y = int(row * tile_h)
                w = int(tile_w) + 1   # "+1" to avoid hairline gaps
                h = int(tile_h) + 1
                rect = pygame.Rect(x, y, w, h)
                # Only draw inside the world area (left side)
                if x < self.world_width:
                    pygame.draw.rect(self.screen, color, rect)

    # ------------------------------------------------------------------ #
    # Trait heatmap rendering
    # ------------------------------------------------------------------ #
    def _draw_heatmap(self) -> None:
        if not self.heatmap_mode:
            return

        trait_key = {
            "met": "met_mean",
            "spd": "spd_mean",
            "sns": "sns_mean",
            "soc": "soc_mean",
            "agg": "agg_mean",
        }[self.heatmap_mode]

        max_val = 0.0
        # find max for normalisation
        for r in range(self.env.rows):
            for c in range(self.env.cols):
                val = self.region_stats.regions[r][c][trait_key]
                if val > max_val:
                    max_val = val

        if max_val <= 0:
            return

        for r in range(self.env.rows):
            for c in range(self.env.cols):
                tile = self.region_stats.regions[r][c]
                val = tile[trait_key]

                if tile["count"] == 0:
                    continue  # no population, leave tile unchanged

                # Normalise 0..1
                t = val / max_val

                # Convert to heat color (blue → red)
                R = int(255 * t)
                G = 0
                B = int(255 * (1 - t))
                color = (R, G, B, 90)  # alpha ~ transparency

                rect = pygame.Rect(
                    c * self.env.tile_w,
                    r * self.env.tile_h,
                    self.env.tile_w,
                    self.env.tile_h,
                )

                # Using a surface for transparency
                s = pygame.Surface((self.env.tile_w, self.env.tile_h), pygame.SRCALPHA)
                s.fill(color)
                self.screen.blit(s, rect)

    def run(self) -> None:
        running = True
        while running:
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        # If brain overlay is open, close it first
                        if self.show_brain_overlay:
                            self.show_brain_overlay = False
                        else:
                            running = False
                    elif event.key == pygame.K_f:
                        self.fast_mode = not self.fast_mode
                    elif event.key == pygame.K_p:
                        self.paused = not self.paused  # PAUSE TOGGLE
                    elif event.key == pygame.K_h:
                        # Cycle heatmap modes
                        modes = [None, "met", "spd", "sns", "soc", "agg"]
                        current = self.heatmap_mode
                        idx = modes.index(current) if current in modes else 0
                        self.heatmap_mode = modes[(idx + 1) % len(modes)]
                    elif event.key == pygame.K_b:
                        # Toggle big brain overlay
                        self.show_brain_overlay = not self.show_brain_overlay
                    elif event.key == pygame.K_r:
                        # Toggle framerate display
                        self.show_framerate = not self.show_framerate

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    self._select_humlet_at(mx, my)

            # ------------------------
            # Update only if not paused
            # ------------------------
            if not self.paused:
                updates_per_frame = 1 if not self.fast_mode else 2
                for _ in range(updates_per_frame):
                    self._update_simulation_step()

            # Draw everything (world, panel, overlays) and flip
            self._draw_frame()

            # Timing
            if not self.fast_mode:
                self.clock.tick(60)
            else:
                self.clock.tick(240)

        pygame.quit()

    # ------------------------------------------------------------------ #
    # Selection
    # ------------------------------------------------------------------ #
    def _select_humlet_at(self, x: int, y: int) -> None:
        """Select the closest humlet within a small radius of the click,
        but only if the click is inside the world area (not the panel)."""
        if x >= self.world_width:
            # Clicked in the panel area – don't change selection
            return

        closest = None
        min_d2 = 15 ** 2
        for h in self.humlets:
            if not h.alive:
                continue
            dx = h.x - x
            dy = h.y - y
            d2 = dx * dx + dy * dy
            if d2 < min_d2:
                min_d2 = d2
                closest = h
        self.selected_humlet = closest

    # ------------------------------------------------------------------ #
    # Simulation step
    # ------------------------------------------------------------------ #
    def _update_simulation_step(self) -> None:
        # Advance simulation time
        self.tick += 1

        # Update environment (time, climate, food respawn)
        self.env.update()
        self._update_population_capacity()

        newborns: list[Humlet] = []

        # Update all humlets
        for h in self.humlets:
            h.update(
                self.env,
                self.humlets,
                newborns,
                max_population=self.max_population,
                spatial_index=self.humlet_index,
            )

        # Remove dead humlets
        self.humlets = [h for h in self.humlets if h.alive]

        # Add newborns and capture their seeds for reproducibility
        self.humlets.extend(newborns)
        for h in newborns:
            self.agent_seeds[h.id] = getattr(h, "seed", None)

        # Update evolution / population statistics
        self.stats.update(self.tick, self.humlets, self.env)

        # Persist evolutionary trajectory if requested
        if self.trajectory_log_path is not None:
            self._append_trajectory_log()

        # Update per-region trait stats for heatmaps
        self._maybe_update_region_stats()

        # Rebuild the spatial index for the next frame
        self._rebuild_humlet_index()

    def _maybe_update_region_stats(self) -> None:
        if self.tick == self._last_region_stats_tick:
            return

        if self.heatmap_mode is None and self.tick % self.region_stats_update_interval != 0:
            return

        self.region_stats.reset()
        for h in self.humlets:
            self.region_stats.accumulate(h)
        self.region_stats.compute_means()
        self._last_region_stats_tick = self.tick

    def _rebuild_humlet_index(self) -> None:
        self.humlet_index.clear()
        for h in self.humlets:
            if h.alive:
                self.humlet_index.insert(h, h.x, h.y)

    # ------------------------------------------------------------------ #
    # Drawing
    # ------------------------------------------------------------------ #
    def _draw_frame(self) -> None:
        # Clear full screen
        self.screen.fill((0, 0, 0))

        # ----------------- WORLD AREA (left side) ----------------- #
        # 1) Biome background (planet patches)
        self._draw_biome_background()
        self._draw_heatmap()

        # 2) World objects on top
        for obj in self.env.objects:
            if isinstance(obj, Food):
                pygame.draw.circle(self.screen, (0, 255, 0), (int(obj.x), int(obj.y)), 3)
            elif isinstance(obj, Rock):
                pygame.draw.circle(self.screen, (120, 120, 120), (int(obj.x), int(obj.y)), 3)
            elif isinstance(obj, Shelter):
                pygame.draw.rect(self.screen, (139, 69, 19), (int(obj.x) - 5, int(obj.y) - 5, 10, 10))
            elif isinstance(obj, Tree):
                pygame.draw.rect(self.screen, (20, 120, 20), (int(obj.x) - 3, int(obj.y) - 3, 6, 6))
            elif isinstance(obj, StoneDeposit):
                pygame.draw.rect(self.screen, (80, 80, 140), (int(obj.x) - 3, int(obj.y) - 3, 6, 6))

        # Draw village centre as a faint ring
        vx = int(self.env.village.x)
        vy = int(self.env.village.y)
        pygame.draw.circle(self.screen, (255, 255, 255), (vx, vy), 10, 1)

        # 3) Humlets
        for humlet in self.humlets:
            if not humlet.alive:
                continue

            # Phenotype-based color
            color = self._phenotype_color(humlet)

            # Mass-based display size
            radius = humlet.render_radius()

            # Screen position
            pos = (int(humlet.x), int(humlet.y))

            # Draw the Humlet
            pygame.draw.circle(self.screen, color, pos, radius)

            # Highlight selected Humlet
        # Draw selection highlight (safe even if humlet list is empty)
        if self.selected_humlet is not None and self.selected_humlet.alive:
            sh = self.selected_humlet
            pos = (int(sh.x), int(sh.y))
            radius = sh.render_radius()

            pygame.draw.circle(
                self.screen, (255, 255, 0),
                pos,
                radius + 4, 2,
            )
            pygame.draw.circle(
                self.screen, (255, 255, 0),
                pos,
                int(sh.sense_range), 1,
            )

            # Optional debugging overlays
            pygame.draw.circle(self.screen, (255, 255, 255), pos, int(sh.sense_range), 1)
            pygame.draw.circle(self.screen, (255, 255, 255), pos, radius + 2, 1)

            v = getattr(sh, "vision", None)
            if v is not None:
                v.draw(self.screen)


        # HUD (top bar, over world area)
        pop = len(self.humlets)
        food_count = sum(1 for obj in self.env.objects if isinstance(obj, Food))
        info = (
            f"Tick: {self.tick}   "
            f"Pop: {pop}   "
            f"Temp: {self.env.temperature:.1f}°C   "
            f"AirQ: {self.env.air_quality:.2f}   "
            f"Mode: {'FAST (F)' if self.fast_mode else 'NORMAL (F)'}"
        )
        text_surface = self.font_medium.render(info, True, (255, 255, 255))
        self.screen.blit(text_surface, (10, 10))

        # Secondary HUD line with environment / ecology context
        day_frac = (self.env.time % max(1, self.env.day_length)) / max(1, self.env.day_length)
        season_frac = (self.env.time % max(1, self.env.season_length)) / max(1, self.env.season_length)
        info2 = (
            f"Day: {day_frac * 100:5.1f}%   "
            f"Season: {season_frac * 100:5.1f}%   "
            f"Light: {self.env.light_level:.2f}   "
            f"Wild food: {food_count}/{self.env.food_capacity}   "
            f"Food pool: {self.env.food_energy_pool:.0f}"
        )
        text_surface2 = self.font_small.render(info2, True, (220, 220, 220))
        self.screen.blit(text_surface2, (10, 32))

        if self.show_framerate:
            fps = self.clock.get_fps()
            fps_text = self.font_small.render(f"FPS: {fps:5.1f} (R)", True, (200, 255, 200))
            self.screen.blit(fps_text, (10, 50))


        # ----------------- INSPECTOR PANEL (right side) ----------------- #
        self._draw_inspector_panel()

        # ----------------- BRAIN OVERLAY (if enabled) ----------------- #
        if self.show_brain_overlay and self.selected_humlet is not None:
            self._draw_brain_overlay(self.selected_humlet)

        # ----------------- PAUSED overlay text ----------------- #
        if self.paused:
            overlay = self.font_medium.render("PAUSED (P)", True, (255, 200, 200))
            self.screen.blit(overlay, (10, 35))

        # --- FINAL: push everything to the screen ---
        pygame.display.flip()      


    # ------------------------------------------------------------------ #
    # Inspector UI (text + brain diagram + bars)
    # ------------------------------------------------------------------ #
    def _draw_inspector_panel(self) -> None:
        panel_width = self.panel_width

        # Two vertical inspector panels
        humlet_rect = pygame.Rect(
            self.world_width + self.margin,
            40,
            panel_width,
            self.world_height - 50,
        )
        pop_rect = pygame.Rect(
            humlet_rect.right + 4,   # small gap between panels
            40,
            panel_width,
            self.world_height - 50,
        )

        # Backgrounds & borders
        for rect in (humlet_rect, pop_rect):
            pygame.draw.rect(self.screen, (15, 15, 25), rect)
            pygame.draw.rect(self.screen, (90, 90, 140), rect, 1)

        # ---------- small helpers ---------- #
        def draw_line(
            text: str,
            x: int,
            y: int,
            color: tuple[int, int, int] = (220, 220, 240),
        ) -> int:
            surf = self.font_small.render(text, True, color)
            self.screen.blit(surf, (x, y))
            return y + 18

        def draw_header(
            text: str,
            x: int,
            y: int,
            color: tuple[int, int, int],
        ) -> int:
            surf = self.font_small.render(text, True, color)
            self.screen.blit(surf, (x, y))
            pygame.draw.line(
                self.screen,
                color,
                (x, y + 15),
                (x + surf.get_width(), y + 15),
                1,
            )
            return y + 22

        def draw_bar(
            label: str,
            frac: float,
            x: int,
            y: int,
            bar_color: tuple[int, int, int],
            back_color: tuple[int, int, int] = (40, 40, 60),
        ) -> int:
            frac = max(0.0, min(1.0, frac))
            bar_x = x
            bar_y = y + 13
            bar_w = panel_width - 16
            bar_h = 8

            # label
            self.screen.blit(
                self.font_small.render(label, True, (210, 210, 210)),
                (x, y),
            )

            # background
            pygame.draw.rect(
                self.screen,
                back_color,
                (bar_x, bar_y, bar_w, bar_h),
            )
            # foreground
            pygame.draw.rect(
                self.screen,
                bar_color,
                (bar_x, bar_y, int(bar_w * frac), bar_h),
            )

            return bar_y + bar_h + 6

        def rel_pct(value: float, avg: float) -> float:
            if avg <= 0:
                return 0.0
            return (value / avg - 1.0) * 100.0

        # ==============================================================
        # LEFT PANEL: Selected Humlet (brain + body + needs + history)
        # ==============================================================

        brain_rect = None
        if self.selected_humlet is not None:
            brain_rect = pygame.Rect(
                humlet_rect.x + 4,
                humlet_rect.y + 4,
                humlet_rect.width - 8,
                170,  # fixed height for the brain diagram
            )
            self._draw_brain_diagram(self.selected_humlet, brain_rect)

        # Starting y for text: below brain if present, otherwise near top
        if brain_rect is not None:
            y = brain_rect.bottom + 8
        else:
            y = humlet_rect.y + 6

        x = humlet_rect.x + 8

        h = self.selected_humlet
        s = self.stats.latest

        if h is not None:
            status_color = (120, 255, 120) if h.alive else (255, 120, 120)
            status = "ALIVE" if h.alive else "DEAD"

            # ID line
            y = draw_header(
                f"Humlet #{h.id}  [{status}]",
                x,
                y,
                status_color,
            )

            # Identity / lineage
            y = draw_line(f"Family: {h.family_id}", x, y)
            y = draw_line(f"Parent: {h.parent_id}", x, y)
            y = draw_line(f"Generation: {h.generation}", x, y)
            y = draw_line(f"Offspring: {getattr(h, 'offspring_count', 0)}", x, y)
            y += 4

            # Group + age
            y = draw_line(f"Group: {h.group_id}", x, y)
            y = draw_line(f"Age: {h.age} / {h.lifespan}", x, y)

            # Life-stage bar (young → old)
            age_frac = min(1.0, h.age / max(1, h.lifespan))
            age_col = (
                int(220 * age_frac),
                int(220 * (1.0 - abs(age_frac - 0.5) * 1.2)),
                80,
            )
            y = draw_bar("Life stage", age_frac, x, y, age_col)
            y += 2

            # --- Reproduction ---
            y = draw_header("--- Reproduction ---", x, y, (255, 200, 200))
            pregnant = getattr(h, "pregnant", False)
            gestation_period = getattr(h, "gestation_period", 0)
            gestation_timer = getattr(h, "gestation_timer", 0)
            repro_cooldown = getattr(h, "repro_cooldown", 0)
            min_energy = getattr(h, "repro_min_energy", 0.0)

            if pregnant and gestation_period > 0:
                progress = 1.0 - (gestation_timer / max(1, gestation_period))
            else:
                progress = 0.0

            y = draw_line(
                f"Status : {'Pregnant' if pregnant else 'Not pregnant'}", x, y
            )
            if pregnant:
                y = draw_line(f"Time left : {gestation_timer} ticks", x, y)
                y = draw_bar("Gestation", progress, x, y, (255, 170, 170))
            y = draw_line(f"Cooldown : {repro_cooldown} ticks", x, y)
            y = draw_line(f"Min energy: {min_energy:.1f}", x, y)
            y += 2

            # --- Physical body ---
            y = draw_header("--- Body ---", x, y, (200, 230, 255))
            y = draw_line(f"Mass   : {getattr(h, 'mass', 0.0):.1f} kg", x, y)
            y = draw_line(f"Height : {getattr(h, 'height', 0.0):.2f} m", x, y)
            y = draw_line(f"Frame  : {getattr(h, 'frame_factor', 1.0):.2f}", x, y)

            dx_home = h.x - getattr(h, "home_x", h.x)
            dy_home = h.y - getattr(h, "home_y", h.y)
            home_dist = (dx_home * dx_home + dy_home * dy_home) ** 0.5
            y = draw_line(f"Home dist: {home_dist:5.1f}", x, y)
            y += 2

            # --- Carrying / resources on person ---
            y = draw_header("--- Inventory ---", x, y, (200, 255, 200))
            y = draw_line(f"Wood carried : {getattr(h, 'carry_wood', 0.0):.1f}", x, y)
            y = draw_line(f"Stone carried: {getattr(h, 'carry_stone', 0.0):.1f}", x, y)
            y += 2

            # Energy & Health bars
            energy_frac = h.energy / max(1e-6, h.max_energy)
            health_frac = h.health / max(1e-6, h.max_health)

            y = draw_bar(
                "Energy",
                energy_frac,
                x,
                y,
                (80, 220, 80) if energy_frac > 0.3 else (220, 120, 80),
            )
            y = draw_bar(
                "Health",
                health_frac,
                x,
                y,
                (80, 220, 80) if health_frac > 0.3 else (220, 120, 80),
            )
            y += 2

            # Needs
            y = draw_header("--- Needs ---", x, y, (180, 220, 255))
            y = draw_line(f"Hunger: {h.hunger_need:.2f}", x, y)
            y = draw_line(f"Safety: {h.safety_need:.2f}", x, y)
            y = draw_line(f"Social: {h.social_need:.2f}", x, y)

            # Brain activity bar (based on last outputs)
            outputs = getattr(h, "last_outputs", None)
            if outputs is not None and len(outputs) > 0:
                act = sum(abs(float(o)) for o in outputs) / len(outputs)
                act_frac = max(0.0, min(1.0, act))
                y = draw_bar("Brain activity", act_frac, x, y, (120, 200, 255))
            y += 4

            # Higher-level
            y = draw_header("--- Higher-level ---", x, y, (255, 210, 120))
            y = draw_line(f"Esteem    : {getattr(h, 'esteem_level', 0.0):.2f}", x, y)
            y = draw_line(f"Curiosity : {getattr(h, 'curiosity_trait', 0.0):.2f}", x, y)
            y = draw_line(f"Cur. drive: {getattr(h, 'curiosity_drive', 0.0):.2f}", x, y)
            y = draw_line(f"Neighbors : {getattr(h, 'neighbor_count', 0)}", x, y)
            y += 4

            # --- Genome with deltas vs population avg ---
            y = draw_header("--- Genome (vs avg) ---", x, y, (200, 255, 180))

            def trait_line(name: str, value: float, avg_val: float) -> int:
                if avg_val > 0:
                    delta = rel_pct(value, avg_val)
                    sign = "+" if delta >= 0 else ""
                    col = (
                        (160, 255, 160)
                        if delta > 5
                        else (255, 160, 160)
                        if delta < -5
                        else (210, 210, 210)
                    )
                    txt = f"{name}: {value:.3f} ({sign}{delta:.0f}%)"
                    return draw_line(txt, x, y, col)
                else:
                    return draw_line(f"{name}: {value:.3f}", x, y)

            if s is not None:
                y = trait_line("Metabolism", h.metabolism_rate, s.avg_metabolism_rate)
                y = trait_line("Speed", h.speed_trait, s.avg_speed)
                y = trait_line("Sense rng", h.sense_range, s.avg_sense_range)
                y = trait_line("Aggression", h.aggression, s.avg_aggression)
                y = trait_line("Sociability", h.sociability, s.avg_sociability)
                y = trait_line("Mass", getattr(h, "mass", 0.0), s.avg_mass)
                y = trait_line("Height", getattr(h, "height", 0.0), s.avg_height)
                y = trait_line("Frame", getattr(h, "frame_factor", 1.0), s.avg_frame_factor)
            else:
                y = draw_line(f"Metabolism: {h.metabolism_rate:.3f}", x, y)
                y = draw_line(f"Speed: {h.speed_trait:.2f}", x, y)
                y = draw_line(f"Sense range: {h.sense_range:.1f}", x, y)
                y = draw_line(f"Aggression: {h.aggression:.2f}", x, y)
                y = draw_line(f"Sociability: {h.sociability:.2f}", x, y)

            y = draw_line(f"Lifespan: {h.lifespan}", x, y)
            y += 4

            # --- Brain meta ---
            y = draw_header("--- Brain ---", x, y, (200, 200, 255))
            y = draw_line(f"W1 shape: {h.brain['W1'].shape}", x, y)
            y = draw_line(f"W2 shape: {h.brain['W2'].shape}", x, y)
            y = draw_line(f"Params : {h.brain_param_count()}", x, y)

        # ==============================================================
        # RIGHT PANEL: Population / evolution / village stats
        # ==============================================================

        if self.stats.latest is not None:
            s = self.stats.latest
            x2 = pop_rect.x + 8
            y2 = pop_rect.y + 6

            y2 = draw_header("== Population Stats ==", x2, y2, (255, 230, 140))
            y2 = draw_line(f"Tick       : {s.tick}", x2, y2)
            y2 = draw_line(f"Population : {s.population}", x2, y2)
            y2 = draw_line(f"Pregnant   : {s.pregnant_count}", x2, y2)
            y2 = draw_line(
                f"Avg gestate: {s.avg_gestation_progress * 100:.1f}%",
                x2,
                y2,
            )
            y2 = draw_line(f"Families   : {s.num_families}", x2, y2)
            y2 = draw_line(f"Max Gen    : {s.max_generation}", x2, y2)
            y2 = draw_line(f"Avg Gen    : {s.avg_generation:.2f}", x2, y2)
            y2 += 4
            y2 = draw_line(f"Avg Speed  : {s.avg_speed:.2f}", x2, y2)
            y2 = draw_line(f"Avg Metab  : {s.avg_metabolism_rate:.3f}", x2, y2)
            y2 = draw_line(f"Avg Sense  : {s.avg_sense_range:.1f}", x2, y2)
            y2 = draw_line(f"Avg Aggress: {s.avg_aggression:.2f}", x2, y2)
            y2 = draw_line(f"Avg Sociab : {s.avg_sociability:.2f}", x2, y2)
            y2 += 4
            y2 = draw_line(f"Avg Energy : {s.avg_energy:.1f}", x2, y2)
            y2 = draw_line(f"Avg Health : {s.avg_health:.1f}", x2, y2)
            y2 += 4

            y2 = draw_header("== Body & Space ==", x2, y2, (200, 230, 255))
            y2 = draw_line(f"Avg Mass   : {s.avg_mass:.1f} kg", x2, y2)
            y2 = draw_line(f"Avg Height : {s.avg_height:.2f} m", x2, y2)
            y2 = draw_line(f"Avg Frame  : {s.avg_frame_factor:.2f}", x2, y2)
            y2 = draw_line(f"Avg Home d.: {s.avg_home_distance:.1f}", x2, y2)
            y2 += 4

            y2 = draw_header("== Carrying ==", x2, y2, (200, 255, 200))
            y2 = draw_line(f"Avg Wood   : {s.avg_carry_wood:.2f}", x2, y2)
            y2 = draw_line(f"Avg Stone  : {s.avg_carry_stone:.2f}", x2, y2)
            y2 += 4

            y2 = draw_header("== Higher-level ==", x2, y2, (255, 210, 160))
            y2 = draw_line(f"Avg Esteem : {s.avg_esteem_level:.2f}", x2, y2)
            y2 = draw_line(f"Avg Curios.: {s.avg_curiosity_trait:.2f}", x2, y2)
            y2 = draw_line(f"Avg Neighb.: {s.avg_neighbors:.2f}", x2, y2)
            y2 += 4

            y2 = draw_header("== Village / Resources ==", x2, y2, (180, 255, 180))
            y2 = draw_line(f"Food total : {s.village_food:.1f}", x2, y2)
            y2 = draw_line(f"Wood total : {s.village_wood:.1f}", x2, y2)
            y2 = draw_line(f"Stone total: {s.village_stone:.1f}", x2, y2)
            y2 += 2
            y2 = draw_line(f"Food / cap : {s.avg_food_per_capita:.2f}", x2, y2)
            y2 = draw_line(f"Wood / cap : {s.avg_wood_per_capita:.2f}", x2, y2)
            y2 = draw_line(f"Stone / cap: {s.avg_stone_per_capita:.2f}", x2, y2)

            y2 += 4
            y2 = draw_header("== Environment ==", x2, y2, (200, 230, 255))
            y2 = draw_line(f"Day phase  : {s.day_progress * 100:5.1f}%", x2, y2)
            y2 = draw_line(f"Season     : {s.season_progress * 100:5.1f}%", x2, y2)
            y2 = draw_line(f"Light lvl  : {s.light_level:.2f}", x2, y2)
            y2 = draw_line(
                f"Wild food  : {s.wild_food_count}/{s.food_capacity}", x2, y2
            )
            y2 = draw_line(f"Food pool  : {s.food_energy_pool:.0f}", x2, y2)
            y2 = draw_line(f"Product/tk : {s.productivity_per_tick:.1f}", x2, y2)




 
    def _draw_brain_overlay(self, h: Humlet) -> None:
        """
        Full-screen brain inspector:
        - Darkened overlay
        - Large 3-layer network diagram
        - Input / output indices and basic weight stats
        - Press B to toggle
        """
        if getattr(h, "last_inputs", None) is None:
            return
        if getattr(h, "last_hidden", None) is None:
            return
        if getattr(h, "last_outputs", None) is None:
            return

        inputs = h.last_inputs
        hidden = h.last_hidden
        outputs = h.last_outputs

        W1 = h.brain["W1"]
        W2 = h.brain["W2"]

        n_in = len(inputs)
        n_hid = len(hidden)
        n_out = len(outputs)

        screen_w, screen_h = self.screen.get_size()

        # --- Dim the whole screen ---
        overlay = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 190))   # translucent black
        self.screen.blit(overlay, (0, 0))

        # --- Outer panel ---
        margin = 30
        panel_rect = pygame.Rect(
            margin,
            margin,
            screen_w - margin * 2,
            screen_h - margin * 2,
        )
        pygame.draw.rect(self.screen, (25, 25, 40), panel_rect)
        pygame.draw.rect(self.screen, (150, 150, 220), panel_rect, 2)

        # Title + hint
        title = self.font_medium.render(
            f"Brain Inspector – Humlet #{h.id}  (B to close)",
            True,
            (240, 240, 255),
        )
        self.screen.blit(title, (panel_rect.x + 12, panel_rect.y + 10))

        # Split panel into left: diagram, right: text
        diagram_rect = pygame.Rect(
            panel_rect.x + 10,
            panel_rect.y + 40,
            int(panel_rect.width * 0.6),
            panel_rect.height - 50,
        )
        info_rect = pygame.Rect(
            diagram_rect.right + 10,
            panel_rect.y + 40,
            panel_rect.right - diagram_rect.right - 20,
            panel_rect.height - 50,
        )

        # Draw big diagram
        self._draw_large_brain_diagram(h, diagram_rect)

        # Draw textual info
        self._draw_brain_text_info(h, inputs, outputs, W1, W2, info_rect)

    def _draw_large_brain_diagram(self, h: Humlet, rect: pygame.Rect) -> None:
        """Bigger, spaced-out brain diagram for the overlay with labeled neurons."""
        inputs = h.last_inputs
        hidden = h.last_hidden
        outputs = h.last_outputs

        W1 = h.brain["W1"]
        W2 = h.brain["W2"]

        n_in = len(inputs)
        n_hid = len(hidden)
        n_out = len(outputs)

        # --- Labels for neurons -----------------------------------------
        input_labels = [
            "0: hunger_need",
            "1: safety_need",
            "2: social_need",
            "3: energy_norm",
            "4: health_norm",
            "5: food_dx",
            "6: food_dy",
            "7: friend_dx",
            "8: friend_dy",
            "9: shelter_dx",
            "10: shelter_dy",
            "11: esteem_level",
            "12: curiosity_drive",
        ]
        # Trim / pad in case brain size changes
        if len(input_labels) < n_in:
            input_labels += [f"{i}: input" for i in range(len(input_labels), n_in)]
        else:
            input_labels = input_labels[:n_in]

        output_labels = ["Move X", "Move Y", "Eat", "Reproduce"]
        if len(output_labels) < n_out:
            output_labels += [f"Out {k}" for k in range(len(output_labels), n_out)]
        else:
            output_labels = output_labels[:n_out]

        # Background
        pygame.draw.rect(self.screen, (35, 35, 55), rect)
        pygame.draw.rect(self.screen, (110, 110, 170), rect, 1)

        # Layout
        x_in = rect.x + 110
        x_hid = rect.x + rect.width // 2
        x_out = rect.right - 110

        def layer_positions(n: int) -> list[int]:
            if n <= 1:
                return [rect.centery]
            top = rect.y + 40
            bottom = rect.bottom - 40
            span = bottom - top
            return [int(top + i * span / max(1, n - 1)) for i in range(n)]

        y_in = layer_positions(n_in)
        y_hid = layer_positions(n_hid)
        y_out = layer_positions(n_out)

        inactive_threshold = 0.08

        def weight_color(w: float) -> tuple[int, int, int]:
            mag = min(1.0, abs(w) / 2.0)
            c = int(80 + 175 * mag)
            return (70, c, 70) if w >= 0 else (c, 70, 70)

        def weight_thickness(w: float) -> int:
            mag = min(1.0, abs(w) / 2.0)
            return max(1, int(1 + 4 * mag))

        def node_color(a: float) -> tuple[int, int, int]:
            a = max(-1.0, min(1.0, float(a)))
            mag = abs(a)
            if mag < inactive_threshold:
                base = int(90 + 80 * (mag / max(inactive_threshold, 1e-6)))
                return (base, base, base)
            if a >= 0:
                g = int(130 + 120 * mag)
                return (60, g, 60)
            else:
                r = int(130 + 120 * mag)
                return (r, 60, 60)

        def draw_inactive_marker(x: int, y: int, active_value: float) -> None:
            if abs(active_value) >= inactive_threshold:
                # subtle glow for active nodes
                halo = min(60, int(abs(active_value) * 90))
                pygame.draw.circle(
                    self.screen,
                    (180 + halo // 2, 180 + halo // 2, 200),
                    (x, y),
                    radius + 3,
                    1,
                )
                return

            pygame.draw.circle(self.screen, (130, 130, 150), (x, y), radius + 3, 2)
            pygame.draw.line(self.screen, (160, 160, 180), (x - 6, y - 6), (x + 6, y + 6), 1)
            pygame.draw.line(self.screen, (160, 160, 180), (x - 6, y + 6), (x + 6, y - 6), 1)

        radius = 8

        # Connections: input -> hidden
        for i in range(n_in):
            for j in range(n_hid):
                w = float(W1[j, i])
                pygame.draw.line(
                    self.screen,
                    weight_color(w),
                    (x_in + 10, y_in[i]),
                    (x_hid - 10, y_hid[j]),
                    weight_thickness(w),
                )

        # Connections: hidden -> output
        for j in range(n_hid):
            for k in range(n_out):
                w = float(W2[k, j])
                pygame.draw.line(
                    self.screen,
                    weight_color(w),
                    (x_hid + 10, y_hid[j]),
                    (x_out - 10, y_out[k]),
                    weight_thickness(w),
                )

        # Input nodes + labels
        for i, y in enumerate(y_in):
            col = node_color(inputs[i])
            pygame.draw.circle(self.screen, col, (x_in, y), radius)
            pygame.draw.circle(self.screen, (230, 230, 230), (x_in, y), radius, 1)
            draw_inactive_marker(x_in, y, inputs[i])

            # label to the left
            label_text = input_labels[i]
            label_surf = self.font_small.render(label_text, True, (230, 230, 230))
            label_x = x_in - 10 - label_surf.get_width()
            self.screen.blit(label_surf, (label_x, y - 7))

            # value to the right for quick inspection
            val_surf = self.font_small.render(f"{inputs[i]:+.2f}", True, (180, 210, 255))
            self.screen.blit(val_surf, (x_in + 14, y - 7))

        # Hidden nodes (just circles for now, no labels to keep it clean)
        for j, y in enumerate(y_hid):
            col = node_color(hidden[j])
            pygame.draw.circle(self.screen, col, (x_hid, y), radius)
            pygame.draw.circle(self.screen, (230, 230, 230), (x_hid, y), radius, 1)
            draw_inactive_marker(x_hid, y, hidden[j])

        # Output nodes + labels
        for k, y in enumerate(y_out):
            col = node_color(outputs[k])
            pygame.draw.circle(self.screen, col, (x_out, y), radius)
            pygame.draw.circle(self.screen, (230, 230, 230), (x_out, y), radius, 1)
            draw_inactive_marker(x_out, y, outputs[k])

            lab = f"{k}: {output_labels[k]}"
            lab_surf = self.font_small.render(lab, True, (230, 230, 230))
            self.screen.blit(lab_surf, (x_out + 15, y - 7))

            val_surf = self.font_small.render(f"{outputs[k]:+.2f}", True, (255, 230, 190))
            self.screen.blit(val_surf, (x_out - 45, y - 7))


    def _draw_brain_text_info(
        self,
        h: Humlet,
        inputs,
        outputs,
        W1,
        W2,
        rect: pygame.Rect,
    ) -> None:
        """Textual info about inputs, outputs and weight stats."""
        pygame.draw.rect(self.screen, (20, 20, 35), rect)
        pygame.draw.rect(self.screen, (100, 100, 150), rect, 1)

        x = rect.x + 8
        y = rect.y + 4
        line_h = 18

        def draw(text, color=(220, 220, 240)):
            nonlocal y
            surf = self.font_small.render(text, True, color)
            self.screen.blit(surf, (x, y))
            y += line_h

        # Inputs
        draw("Inputs (index : value)", (180, 220, 255))
        for i, val in enumerate(inputs):
            if y > rect.bottom - 3 * line_h:
                break
            draw(f"{i:02d}: {val:+.3f}")

        y += line_h // 2
        draw("Outputs", (255, 220, 160))
        labels = ["Move X", "Move Y", "Eat", "Reproduce"]
        for k, val in enumerate(outputs):
            if y > rect.bottom - 3 * line_h:
                break
            lab = labels[k] if k < len(labels) else f"Out {k}"
            draw(f"{k}: {lab:10s} = {val:+.3f}")

        # Basic weight stats
        import numpy as np

        y += line_h // 2
        draw(
            "Dormant nodes (<0.05 mag)",
            (200, 210, 255),
        )
        hidden = getattr(h, "last_hidden", np.zeros(len(W1)))
        draw(
            f"in:{int(np.sum(np.abs(inputs) < 0.05))}/{len(inputs)}  "
            f"hid:{int(np.sum(np.abs(hidden) < 0.05))}/{len(hidden)}  "
            f"out:{int(np.sum(np.abs(outputs) < 0.05))}/{len(outputs)}",
            (190, 200, 230),
        )

        y += line_h // 2
        draw("W1 stats", (200, 255, 200))
        draw(f"min: {float(np.min(W1)):+.3f}")
        draw(f"max: {float(np.max(W1)):+.3f}")
        draw(f"mean: {float(np.mean(W1)):+.3f}")

        y += line_h // 2
        draw("W2 stats", (200, 255, 200))
        draw(f"min: {float(np.min(W2)):+.3f}")
        draw(f"max: {float(np.max(W2)):+.3f}")
        draw(f"mean: {float(np.mean(W2)):+.3f}")

        y += line_h // 2
        draw(f"Last reward: {h.last_brain_reward:+.3f}", (255, 205, 180))
        draw(f"Fitness score: {h.brain_fitness:+.3f}", (255, 205, 180))

        y += line_h // 2
        draw("Hint: B = toggle overlay", (180, 200, 255))



    # ------------------------------------------------------------------ #
    # Brain diagram rendering
    # ------------------------------------------------------------------ #
    def _draw_brain_diagram(self, h: Humlet, rect: pygame.Rect) -> None:
        """Draw a simple 3-layer brain diagram (inputs, hidden, outputs)
        for the selected humlet, using last activations & weights."""
        # Need activations to show anything
        if getattr(h, "last_inputs", None) is None:
            return

        inputs = h.last_inputs
        hidden = getattr(h, "last_hidden", None)
        outputs = getattr(h, "last_outputs", None)

        if hidden is None or outputs is None:
            return

        W1 = h.brain["W1"]
        W2 = h.brain["W2"]

        n_in = len(inputs)
        n_hid = len(hidden)
        n_out = len(outputs)

        # Panel background
        pygame.draw.rect(self.screen, (30, 30, 40), rect)
        pygame.draw.rect(self.screen, (120, 120, 160), rect, 1)

        title = self.font_small.render("Brain Diagram", True, (240, 240, 255))
        self.screen.blit(title, (rect.x + 6, rect.y + 4))

        # Horizontal positions for each layer
        x_in = rect.x + 35
        x_hid = rect.x + rect.width // 2
        x_out = rect.x + rect.width - 35

        # Vertical spacing helper
        def layer_positions(n: int) -> list[int]:
            if n <= 1:
                return [rect.centery]
            top = rect.y + 25
            bottom = rect.bottom - 20
            span = bottom - top
            return [int(top + i * span / (n - 1)) for i in range(n)]

        y_in = layer_positions(n_in)
        y_hid = layer_positions(n_hid)
        y_out = layer_positions(n_out)

        # Color helpers
        inactive_threshold = 0.08

        def weight_color(w: float) -> tuple[int, int, int]:
            mag = min(1.0, abs(w) / 2.0)
            c = int(80 + 175 * mag)
            if w >= 0:
                return (60, c, 60)   # positive = greenish
            else:
                return (c, 60, 60)   # negative = reddish

        def weight_thickness(w: float) -> int:
            mag = min(1.0, abs(w) / 2.0)
            return max(1, int(1 + 3 * mag))

        def node_color(a: float) -> tuple[int, int, int]:
            # a is typically in [-1, 1]
            a = max(-1.0, min(1.0, float(a)))
            mag = abs(a)
            if mag < inactive_threshold:
                base = int(80 + 90 * (mag / max(1e-6, inactive_threshold)))
                return (base, base, base)
            if a >= 0:
                g = int(120 + 120 * mag)
                return (50, g, 50)
            else:
                r = int(120 + 120 * mag)
                return (r, 50, 50)

        def draw_inactive_marker(x: int, y: int, active_value: float) -> None:
            if abs(active_value) < inactive_threshold:
                pygame.draw.circle(self.screen, (110, 110, 130), (x, y), radius + 2, 1)
                pygame.draw.line(self.screen, (130, 130, 150), (x - 4, y - 4), (x + 4, y + 4), 1)
                pygame.draw.line(self.screen, (130, 130, 150), (x - 4, y + 4), (x + 4, y - 4), 1)
            else:
                halo = min(50, int(abs(active_value) * 80))
                pygame.draw.circle(self.screen, (180, 180, 200), (x, y), radius + 2, 1)

        radius = 5

        # ---- Draw connections: inputs -> hidden ----
        for i in range(n_in):
            for j in range(n_hid):
                w = float(W1[j, i])
                color = weight_color(w)
                thick = weight_thickness(w)
                pygame.draw.line(
                    self.screen,
                    color,
                    (x_in + 6, y_in[i]),
                    (x_hid - 6, y_hid[j]),
                    thick,
                )

        # ---- Draw connections: hidden -> outputs ----
        for j in range(n_hid):
            for k in range(n_out):
                w = float(W2[k, j])
                color = weight_color(w)
                thick = weight_thickness(w)
                pygame.draw.line(
                    self.screen,
                    color,
                    (x_hid + 6, y_hid[j]),
                    (x_out - 6, y_out[k]),
                    thick,
                )

        # ---- Draw nodes ----
        # Input nodes
        for i, y in enumerate(y_in):
            col = node_color(inputs[i])
            pygame.draw.circle(self.screen, col, (x_in, y), radius)
            pygame.draw.circle(self.screen, (220, 220, 220), (x_in, y), radius, 1)
            draw_inactive_marker(x_in, y, inputs[i])

        # Hidden nodes
        for j, y in enumerate(y_hid):
            col = node_color(hidden[j])
            pygame.draw.circle(self.screen, col, (x_hid, y), radius)
            pygame.draw.circle(self.screen, (220, 220, 220), (x_hid, y), radius, 1)
            draw_inactive_marker(x_hid, y, hidden[j])

        # Output nodes + small labels
        output_labels = ["Mx", "My", "Eat", "Repr"]
        for k, y in enumerate(y_out):
            col = node_color(outputs[k])
            pygame.draw.circle(self.screen, col, (x_out, y), radius)
            pygame.draw.circle(self.screen, (220, 220, 220), (x_out, y), radius, 1)
            draw_inactive_marker(x_out, y, outputs[k])

            if k < len(output_labels):
                lab = self.font_small.render(output_labels[k], True, (230, 230, 230))
                self.screen.blit(lab, (x_out + 8, y - 7))

