# village.py
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Village:
    """
    Very simple colony state for Phase 1:
    - position (centre of village)
    - stored resources (food/wood/stone)
    """
    x: float
    y: float
    stored_food: float = 0.0
    stored_wood: float = 0.0
    stored_stone: float = 0.0

    def add_resource(self, kind: str, amount: float) -> None:
        if amount <= 0:
            return
        if kind == "food":
            self.stored_food += amount
        elif kind == "wood":
            self.stored_wood += amount
        elif kind == "stone":
            self.stored_stone += amount

    def totals(self) -> dict:
        return {
            "food": self.stored_food,
            "wood": self.stored_wood,
            "stone": self.stored_stone,
        }
