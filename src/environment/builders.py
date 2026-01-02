import numpy as np
from typing import Tuple, Optional
from .grid_world import GridWorld


class BuilderAgent:
    """Builder agent that executes construction instructions with variable competence."""
    
    def __init__(self, builder_id: int, competence: float = 0.9, grid: GridWorld = None):
        self.builder_id = builder_id
        self.competence = competence
        self.grid = grid
        self.position = (0, 0)
        self.execution_time = int(1.0 / competence)
        self.steps_until_ready = 0
        
    def set_grid(self, grid: GridWorld):
        """Set the grid this builder operates on."""
        self.grid = grid
        
    def reset(self):
        """Reset builder state."""
        self.position = (0, 0)
        self.steps_until_ready = 0
        
    def is_ready(self) -> bool:
        """Check if builder is ready to execute new instruction."""
        return self.steps_until_ready <= 0
        
    def step(self):
        """Advance one time step."""
        if self.steps_until_ready > 0:
            self.steps_until_ready -= 1
            
    def execute_instruction(self, instruction: dict) -> Tuple[bool, str]:
        """
        Execute a construction instruction.
        
        Args:
            instruction: Dict with keys 'action', 'color', 'x', 'y'
            
        Returns:
            (success, message) tuple
        """
        if not self.is_ready():
            return False, "Builder is busy"
            
        if self.grid is None:
            return False, "No grid assigned"
            
        action = instruction.get('action')
        
        if action == 'place':
            return self._execute_place(
                instruction.get('color', 0),
                instruction.get('x', 0),
                instruction.get('y', 0)
            )
        elif action == 'move':
            return self._execute_move(
                instruction.get('x', 0),
                instruction.get('y', 0)
            )
        elif action == 'wait':
            return self._execute_wait()
        else:
            return False, f"Unknown action: {action}"
            
    def _execute_place(self, color: int, x: int, y: int) -> Tuple[bool, str]:
        """Execute a place block instruction with possible failures."""
        self.steps_until_ready = self.execution_time
        
        success_roll = np.random.random()
        if success_roll > self.competence:
            failure_type = np.random.choice(['wrong_color', 'wrong_position', 'no_action'])
            
            if failure_type == 'wrong_color':
                wrong_color = (color + np.random.randint(1, 4)) % 4
                self.grid.place_block(wrong_color, x, y)
                return False, f"Placed wrong color {wrong_color} instead of {color}"
                
            elif failure_type == 'wrong_position':
                wrong_x = max(0, min(self.grid.grid_size - 1, x + np.random.randint(-1, 2)))
                wrong_y = max(0, min(self.grid.grid_size - 1, y + np.random.randint(-1, 2)))
                self.grid.place_block(color, wrong_x, wrong_y)
                return False, f"Placed at wrong position ({wrong_x}, {wrong_y})"
                
            else:
                return False, "Failed to execute action"
        
        success = self.grid.place_block(color, x, y)
        if success:
            self.position = (x, y)
            return True, f"Placed color {color} at ({x}, {y})"
        else:
            return False, "Invalid placement"
            
    def _execute_move(self, x: int, y: int) -> Tuple[bool, str]:
        """Execute a move instruction."""
        self.steps_until_ready = max(1, self.execution_time // 2)
        
        success_roll = np.random.random()
        if success_roll > self.competence:
            return False, "Failed to move"
            
        if 0 <= x < self.grid.grid_size and 0 <= y < self.grid.grid_size:
            self.position = (x, y)
            return True, f"Moved to ({x}, {y})"
        else:
            return False, "Invalid position"
            
    def _execute_wait(self) -> Tuple[bool, str]:
        """Execute a wait instruction."""
        self.steps_until_ready = 1
        return True, "Waiting"


class BuilderTeam:
    """Manages a team of builders with shared competence level."""
    
    def __init__(self, team_id: str, num_builders: int, competence: float, grid: GridWorld):
        self.team_id = team_id
        self.competence = competence
        self.builders = [
            BuilderAgent(i, competence, grid) 
            for i in range(num_builders)
        ]
        
    def reset(self):
        """Reset all builders in team."""
        for builder in self.builders:
            builder.reset()
            
    def step(self):
        """Advance all builders one time step."""
        for builder in self.builders:
            builder.step()
            
    def get_available_builder(self) -> Optional[BuilderAgent]:
        """Get first available builder, or None if all busy."""
        for builder in self.builders:
            if builder.is_ready():
                return builder
        return None
        
    def execute_instruction(self, instruction: dict) -> Tuple[bool, str]:
        """Execute instruction with first available builder."""
        builder = self.get_available_builder()
        if builder is None:
            return False, "All builders busy"
        return builder.execute_instruction(instruction)
        
    def get_status(self) -> dict:
        """Get team status summary."""
        ready_count = sum(1 for b in self.builders if b.is_ready())
        return {
            'team_id': self.team_id,
            'competence': self.competence,
            'total_builders': len(self.builders),
            'ready_builders': ready_count,
            'busy_builders': len(self.builders) - ready_count
        }
