from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from collections import deque

# Grid dimensions
GRID_WIDTH = 30
GRID_HEIGHT = 20

# City dimensions (each city is 5x5)
CITY_SIZE = 5


class CellType(Enum):
    PATH = "path"
    BLOCKED = "blocked"
    CITY = "city"


@dataclass
class Grid:
    width: int
    height: int
    cells: List[List[CellType]] = field(default_factory=list)
    city_positions: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    def __post_init__(self):
        # Initialize grid with all paths
        self.cells = [[CellType.PATH for _ in range(self.width)] for _ in range(self.height)]

        # Place cities (each city is 5x5)
        # City of Seeds at top-left area
        self.city_positions['Seeds'] = (3, 3)
        # City of Mulch at top-right area
        self.city_positions['Mulch'] = (22, 3)
        # City of Harvest at bottom-center area
        self.city_positions['Harvest'] = (12, 13)

        # Mark city cells
        for city_name, (cx, cy) in self.city_positions.items():
            for dy in range(CITY_SIZE):
                for dx in range(CITY_SIZE):
                    if 0 <= cx + dx < self.width and 0 <= cy + dy < self.height:
                        self.cells[cy + dy][cx + dx] = CellType.CITY

        # Add some blocked cells around the map for obstacles
        # Create some natural barriers
        blocked_regions = [
            # Mountains between cities
            (10, 8), (11, 8), (12, 8), (13, 8), (14, 8),
            (15, 9), (16, 9), (17, 9),
            # Forest areas
            (1, 10), (2, 10), (1, 11), (2, 11),
            (26, 10), (27, 10), (28, 10), (26, 11), (27, 11),
        ]

        for bx, by in blocked_regions:
            if 0 <= bx < self.width and 0 <= by < self.height:
                # Don't block city cells
                if self.cells[by][bx] != CellType.CITY:
                    self.cells[by][bx] = CellType.BLOCKED

    def get_city_at(self, x: int, y: int) -> Optional[str]:
        """Returns the city name if the position is within a city, None otherwise"""
        for city_name, (cx, cy) in self.city_positions.items():
            if cx <= x < cx + CITY_SIZE and cy <= y < cy + CITY_SIZE:
                return city_name
        return None

    def is_valid_move(self, x: int, y: int) -> bool:
        """Check if a position is valid for movement"""
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        return self.cells[y][x] != CellType.BLOCKED

    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Get valid neighboring positions (up, down, left, right)"""
        neighbors = []
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if self.is_valid_move(nx, ny):
                neighbors.append((nx, ny))

        return neighbors

    def find_path_to_city(self, start_x: int, start_y: int, target_city: str) -> List[Tuple[int, int]]:
        """Find shortest path from start position to target city using BFS"""
        if target_city not in self.city_positions:
            return []

        # Get city center as target
        cx, cy = self.city_positions[target_city]
        target_x, target_y = cx + 2, cy + 2  # Center of 5x5 city

        # If already at target
        if start_x == target_x and start_y == target_y:
            return []

        # BFS to find shortest path
        queue = deque([(start_x, start_y, [])])
        visited = {(start_x, start_y)}

        while queue:
            x, y, path = queue.popleft()

            # Check all neighbors
            for nx, ny in self.get_neighbors(x, y):
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    new_path = path + [(nx, ny)]

                    # Check if we reached the target city
                    if self.get_city_at(nx, ny) == target_city:
                        return new_path

                    queue.append((nx, ny, new_path))

        return []  # No path found
