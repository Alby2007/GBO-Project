import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional


class GridWorld:
    """Simple grid-based construction environment."""
    
    def __init__(self, grid_size: int = 10, max_colors: int = 4):
        self.grid_size = grid_size
        self.max_colors = max_colors
        self.grid = np.zeros((grid_size, grid_size), dtype=np.int32)
        
    def reset(self):
        """Clear the grid."""
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        
    def place_block(self, color: int, x: int, y: int) -> bool:
        """Place a colored block at position (x, y)."""
        if not self._is_valid_position(x, y):
            return False
        if color < 0 or color >= self.max_colors:
            return False
        self.grid[y, x] = color + 1
        return True
        
    def get_block(self, x: int, y: int) -> int:
        """Get color at position (x, y). Returns 0 for empty."""
        if not self._is_valid_position(x, y):
            return 0
        return self.grid[y, x]
        
    def _is_valid_position(self, x: int, y: int) -> bool:
        """Check if position is within grid bounds."""
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size
        
    def get_state(self) -> np.ndarray:
        """Return current grid state."""
        return self.grid.copy()
        
    def matches_pattern(self, pattern: np.ndarray, offset_x: int = 0, offset_y: int = 0) -> bool:
        """Check if a pattern matches at given offset."""
        pattern_h, pattern_w = pattern.shape
        if offset_x + pattern_w > self.grid_size or offset_y + pattern_h > self.grid_size:
            return False
        
        grid_section = self.grid[offset_y:offset_y+pattern_h, offset_x:offset_x+pattern_w]
        return np.array_equal(grid_section, pattern)


class Task:
    """Represents a construction task with a target pattern and deadline."""
    
    def __init__(self, task_id: int, target_pattern: np.ndarray, deadline: int):
        self.task_id = task_id
        self.target_pattern = target_pattern
        self.deadline = deadline
        self.start_time = 0
        self.completion_time = None
        self.completed = False
        
    def check_completion(self, grid: GridWorld, current_time: int) -> bool:
        """Check if task is completed in the grid."""
        pattern_h, pattern_w = self.target_pattern.shape
        
        for y in range(grid.grid_size - pattern_h + 1):
            for x in range(grid.grid_size - pattern_w + 1):
                if grid.matches_pattern(self.target_pattern, x, y):
                    if not self.completed:
                        self.completed = True
                        self.completion_time = current_time
                    return True
        return False
        
    def is_on_time(self, current_time: int) -> bool:
        """Check if task was completed on time."""
        if not self.completed:
            return False
        return self.completion_time <= self.deadline
        
    def is_late(self, current_time: int) -> bool:
        """Check if task is late but within grace period (2x deadline)."""
        if not self.completed:
            return False
        return self.deadline < self.completion_time <= 2 * self.deadline
        
    def is_failed(self, current_time: int) -> bool:
        """Check if task has failed (missed deadline completely)."""
        if self.completed:
            return self.completion_time > 2 * self.deadline
        return current_time > 2 * self.deadline


def generate_simple_task(task_id: int, complexity: str = "simple", deadline: int = 100) -> Task:
    """Generate a simple construction task."""
    if complexity == "simple":
        pattern = np.array([
            [1, 2],
            [2, 1]
        ])
    elif complexity == "medium":
        pattern = np.array([
            [1, 2, 1],
            [2, 3, 2],
            [1, 2, 1]
        ])
    elif complexity == "hard":
        pattern = np.array([
            [1, 2, 3, 2],
            [2, 3, 1, 3],
            [3, 1, 2, 1],
            [2, 3, 2, 1]
        ])
    else:
        pattern = np.array([[1, 2], [2, 1]])
        
    return Task(task_id, pattern, deadline)
