# sensors/hearing.py
import math

class Hearing:
    """
    Very simple sound sensing system.
    Detects nearby movement or fighting events.
    """

    def __init__(self, owner, hearing_range=150):
        self.owner = owner
        self.range = hearing_range

    def sense(self, environment):
        ox, oy = self.owner.x, self.owner.y

        loudness = 0.0

        for event in environment.sound_events:
            dx = event.x - ox
            dy = event.y - oy
            dist = math.hypot(dx, dy)
            if dist < self.range:
                loudness += (self.range - dist) / self.range

        return loudness
