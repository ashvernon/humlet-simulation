# spatial_hash.py
from __future__ import annotations
from typing import List, Tuple, Set
from collections import defaultdict

class SpatialHash:
    """
    Spatial hash grid for fast neighbor queries.
    Reduces O(n²) to O(n) for proximity checks.
    """
    
    def __init__(self, world_width: float, world_height: float, cell_size: float = 100.0):
        self.width = world_width
        self.height = world_height
        self.cell_size = cell_size
        self.grid: dict[Tuple[int, int], List] = defaultdict(list)
        
    def clear(self):
        """Clear all cells."""
        self.grid.clear()
        
    def _get_cell(self, x: float, y: float) -> Tuple[int, int]:
        """Get the cell coordinates for a position."""
        return (int(x / self.cell_size), int(y / self.cell_size))
    
    def insert(self, obj, x: float, y: float):
        """Insert an object at position (x, y)."""
        cell = self._get_cell(x, y)
        self.grid[cell].append(obj)
    
    def query_radius(self, x: float, y: float, radius: float) -> List:
        """
        Get all objects within radius of (x, y).
        Only checks nearby cells, not the entire world.
        """
        results = []
        
        # Determine which cells to check
        cell_radius = int(radius / self.cell_size) + 1
        center_cell = self._get_cell(x, y)
        
        # Check neighboring cells
        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                cell = (center_cell[0] + dx, center_cell[1] + dy)
                if cell in self.grid:
                    results.extend(self.grid[cell])
        
        return results
    
    def query_radius_wrapped(self, x: float, y: float, radius: float, 
                            env_width: float, env_height: float) -> List:
        """
        Query with toroidal world wrapping.
        Handles edge cases where neighbors might wrap around.
        """
        results = []
        cell_radius = int(radius / self.cell_size) + 1
        center_cell = self._get_cell(x, y)
        
        cols = int(self.width / self.cell_size) + 1
        rows = int(self.height / self.cell_size) + 1
        
        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                # Wrap cell coordinates
                cell_x = (center_cell[0] + dx) % cols
                cell_y = (center_cell[1] + dy) % rows
                cell = (cell_x, cell_y)
                
                if cell in self.grid:
                    results.extend(self.grid[cell])
        
        return results




