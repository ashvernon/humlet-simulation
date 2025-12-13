# Humlet Simulation Feature Checklist

A staged checklist of features and design constraints for growing the Humlet simulation from ecological survival toward civilization-like dynamics.

## Core evolutionary prerequisites
- [x] Scarcity-driven tradeoffs: resources limited; meaningful opportunity costs for choices.
- [x] Heritable variation: genes, cultural traits, or learned policies persisting across generations.
- [ ] Knowledge transmission: imitation, teaching, writing, or institutional memory to retain discoveries.
- [ ] Composable actions: primitives that chain (e.g., gather → craft → store → build → defend → coordinate).
- [x] Spatial structure: toroidal grid or equivalent with travel cost to create "near" vs. "far" dynamics.

## Staged complexity roadmap
- [x] **Stage 1 – Ecology & niches**: food patches, seasons, hazards, shelter value, movement cost; optional predation/disease; real scarcity.
- [x] **Stage 2 – Reproduction & selection stability**: maintain churn without extinction or immortal explosion; trait tradeoffs (metabolism/speed/senses) influence survival.
- [ ] **Stage 3 – Social behaviors with payoff**: cooperation grants advantages (shared defense, better hunting, childcare, pooled resources).
- [ ] **Stage 4 – Culture/memes**: groups hold belief parameters (aggression, sharing, exploration); biased imitation of successful neighbors.
- [ ] **Stage 5 – Technology as capabilities**: discoveries, knowledge spread, infrastructure persistence; tokens like storage, cooking, farming, walls, boats, roads, weapons, medicine (each adds parameter shifts, new actions, and buildings).
- [ ] **Stage 6 – War as economics**: uneven resources + groups + travel + payoff for aggression; territorial claims, raids/defense, revenge/reputation enabling larger conflicts.

## Capability-based ABM pattern
- [x] Agents have needs, senses, and bounded policies (NN/rules).
- [ ] Agents belong to a band/tribe with shared store, cultural parameters, tech set, and territory influence map.
- [ ] Tech unlocks change feasible actions and efficiency rather than raw intelligence.

## Near-term upgrades for current project
- [ ] Introduce tribes/groups with membership via nearest hub or inheritance.
- [ ] Implement shared stores (calories, food items, wood, stone) and simple probabilistic roles (forager/builder/defender).
- [ ] Add persistent buildings (camp, storage, wall, farm plot) that modify local survival odds or yields.
- [ ] Enable cultural imitation: tribes carry 3–6 norms and occasionally adopt neighbors' norms based on success.
- [ ] Build tech as new building types + recipes unlocked via discovered capabilities.

## Common pitfalls to avoid
- [ ] Over-intelligent agents in under-structured worlds—ensure strong constraints first.
- [ ] Excess realism early—prioritize believable incentives over detailed physics/biology.
