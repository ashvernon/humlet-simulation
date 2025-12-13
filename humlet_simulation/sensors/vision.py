# sensors/vision.py
import math
import pygame

class Vision:
    """
    Human-like directional vision with a forward cone and multiple rays.
    """
    def __init__(self, owner, fov_deg=120, num_rays=12, vision_range=120):
        self.owner = owner
        self.fov = math.radians(fov_deg)
        self.num_rays = num_rays
        self.range = vision_range

    def sense(self, environment):
        """
        Returns a list of distances and object codes for each ray.
        """
        ox, oy = self.owner.x, self.owner.y
        direction = self.owner.direction

        results = []

        start_angle = direction - self.fov / 2
        delta = self.fov / self.num_rays

        for i in range(self.num_rays):
            ang = start_angle + i * delta
            dx = math.cos(ang)
            dy = math.sin(ang)

            hit_type = 0
            hit_dist = self.range

            # Check collisions with environment objects
            for obj in environment.objects:
                px, py = obj.x, obj.y
                vx = px - ox
                vy = py - oy

                proj = vx * dx + vy * dy
                if 0 < proj < self.range:
                    # perpendicular distance for ray intersection approx
                    perp = abs(vx * dy - vy * dx)
                    if perp < obj.size:  # object radius
                        hit_dist = proj
                        hit_type = obj.obj_type

            results.append((hit_dist, hit_type))

        return results

    def draw(self, screen):
        """Draws raycasts for debugging."""
        ox, oy = self.owner.x, self.owner.y
        start_angle = self.owner.direction - self.fov / 2
        delta = self.fov / self.num_rays

        for i in range(self.num_rays):
            ang = start_angle + i * delta
            ex = ox + math.cos(ang) * self.range
            ey = oy + math.sin(ang) * self.range
            pygame.draw.line(screen, (60, 60, 255), (ox, oy), (ex, ey), 1)
