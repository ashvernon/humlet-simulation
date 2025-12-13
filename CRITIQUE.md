# Humlet Simulation: Top 10 scientific and engineering risks

1. **Simplistic collision resolution** – Agents now repel from neighbours and solid objects instead of ghosting through them, but contacts are purely kinematic (no momentum, friction, or injury), so crowd pressure and blocking are limited to overlap avoidance rather than richer physical interactions. 【F:humlet_simulation/humlet.py†L313-L414】【F:humlet_simulation/humlet.py†L937-L959】

2. **Static, hand-tuned energy economics** – Metabolism, movement cost, and eating are arbitrary constants with no linkage to body mass, activity duration, or temperature, so survival pressure is artificial and not comparable across configurations. For example, speed directly sets velocity but does not scale metabolic burn or fatigue. 【F:humlet_simulation/humlet.py†L858-L892】

3. **Unbounded food import and no mass conservation** – Food respawns deterministically every few ticks up to a fixed cap regardless of consumption or biome productivity, meaning energy appears from nowhere and population dynamics are decoupled from ecological carrying capacity. 【F:humlet_simulation/environment.py†L117-L135】

4. **Reproduction without gestation or parental cost** – Offspring are created instantly with only a 35% energy penalty and no pregnancy time, investment, or survival trade-offs, making evolutionary signals weak and enabling unrealistically rapid clonal expansion. 【F:humlet_simulation/humlet.py†L724-L783】

5. **No genetic linkage between morphology and performance** – Mass and height traits are sampled and rendered but never influence movement, sensing, injury, or resource needs, so the added "physical" parameters do not feed back into fitness and reduce the scientific validity of the model. 【F:humlet_simulation/humlet.py†L169-L200】【F:humlet_simulation/humlet.py†L858-L892】

6. **Perception ignores occlusion and wrapping edge cases** – Smell/vision vectors simply average world object offsets without accounting for toroidal wrap distances or obstacles, so sensory inputs can point through the planet and double-count distant items, biasing behaviour. 【F:humlet_simulation/sensors/smell.py†L10-L44】【F:humlet_simulation/humlet.py†L800-L823】

7. **Biomes cosmetic only** – Biome attributes (temperature offsets, humidity, fertility) are generated and drawn, but they barely affect food respawn, health, movement, or reproduction; climate cycles update temperature but no heat stress or shelter benefits apply, so "environmental" complexity is largely visual. 【F:humlet_simulation/environment.py†L96-L186】【F:humlet_simulation/environment.py†L108-L135】

8. **Neural controller not trained or grounded** – The feedforward network weights are random and mutated but never evaluated against explicit rewards; movement and feeding thresholds are hard-coded, so evolution lacks selective pressure beyond survival of arbitrary settings, making the brains decorative rather than validated. 【F:humlet_simulation/humlet.py†L809-L884】【F:humlet_simulation/humlet.py†L700-L783】

9. **Population limits detached from resources** – Carrying capacity is a fixed `max_population = num_humlets * 3`, not derived from energy flux or habitat size, so demographic outcomes are clamped by an external cap instead of emergent ecological limits. 【F:humlet_simulation/simulation.py†L22-L74】【F:humlet_simulation/humlet.py†L724-L783】

10. **No stochastic validation or repeatability** – The simulation seeds randomness differently per agent but never exposes reproducible seeds or logging of evolutionary trajectories; results cannot be replicated or statistically compared, undermining any scientific claims. 【F:humlet_simulation/humlet.py†L87-L200】【F:humlet_simulation/simulation.py†L12-L80】

