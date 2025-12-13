# sensors/smell.py
import math

class Smell:
    """
    360° omnidirectional scent sensing.
    Returns vector toward the strongest food smell.
    """

    def __init__(self, owner, smell_range: float = 80.0, sensitivity=1.0):
        self.owner = owner
        self.range = smell_range
        self.sensitivity = sensitivity

    def sense(self, environment):
        ox, oy = self.owner.x, self.owner.y

        fx = fy = 0.0  # scent direction vector

        for obj in environment.objects:
            # Use the world's string "type" instead of a numeric obj_type
            if getattr(obj, "type", None) != "food":
                continue

            dx = obj.x - ox
            dy = obj.y - oy
            dist = math.hypot(dx, dy)

            # Ignore anything outside smell range or at exact same location
            if dist <= 1e-6 or dist > self.range:
                continue

            # Stronger weight for closer food
            strength = (self.range - dist) / self.range
            fx += dx * strength * self.sensitivity
            fy += dy * strength * self.sensitivity

        # Normalise to a unit-ish direction vector the brain can consume
        mag = math.hypot(fx, fy)
        if mag < 1e-6:
            return (0.0, 0.0)

        return (fx / mag, fy / mag)

