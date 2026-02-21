import numpy as np
from sphere_exploration.max_min_on_sphere import ParameterSpaceExplorer


class NpVectorized(ParameterSpaceExplorer):
    def compute(
        self, center: np.ndarray, server_estimate: np.ndarray, radius: int, points: int = 11
    ) -> tuple[float, float]:
        dim_size: int = center.size
        radius_sq: int = radius ** 2

        # Create a tensor with all possible combinations of points
        ranges: np.ndarray = np.array([np.linspace(c - radius, c + radius, points) for c in center])
        coords: np.ndarray = np.array(np.meshgrid(*ranges, indexing="ij")).reshape(dim_size, -1).T

        # Calculate the squared distances from the center
        squared_dists: np.ndarray = np.sum((coords - center) ** 2, axis=1)

        # Get the valid coordinates with squared distances <= radius_sq
        valid_coords: np.ndarray = coords[squared_dists <= radius_sq]

        # Calculate the L-infinity norm between valid_coords and server_estimate
        vals: np.ndarray = np.linalg.norm(valid_coords - server_estimate, ord=np.inf, axis=1)

        # Get min and max values
        tmp_min: float = np.min(vals)
        tmp_max: float = np.max(vals)

        return tmp_min, tmp_max
