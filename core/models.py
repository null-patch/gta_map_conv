from typing import Tuple, List, Dict, Any
from dataclasses import dataclass

@dataclass
class SceneObject:
    """Represents a placed object in the scene"""
    id: int
    model_name: str
    model_data: Dict[str, Any]
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float]
    scale: Tuple[float, float, float]
    flags: int
    draw_distance: float
    texture_dict: str
    lod_level: int = 0
    parent_id: int = -1
    time_on: int = 0
    time_off: int = 24
    interior: int = 0

    def get_transform_matrix(self) -> List[List[float]]:
        return [
            [self.scale[0], 0, 0, self.position[0]],
            [0, self.scale[1], 0, self.position[1]],
            [0, 0, self.scale[2], self.position[2]],
            [0, 0, 0, 1],
        ]
