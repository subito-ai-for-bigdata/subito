import numpy as np
from sphere_exploration.max_min_on_sphere import ParameterSpaceExplorer


class Standard(ParameterSpaceExplorer):
    def compute(
        self, center: np.ndarray, server_estimate: np.ndarray, radius: int, points: int = 2
    ) -> tuple[float, float]:
        tmp_min, tmp_max = float("inf"), float("-inf")
        dim_size: int = center.size
        radius_sq: float = pow(radius, 2)
        coords: np.ndarray = np.zeros(dim_size, dtype=np.float32)
        all_points: int = pow(points, dim_size)

        for i in range(all_points):
            z: int = i
            dist: int = 0

            for j in range(0, dim_size):
                prod: float = (z % points) / (points - 1)
                coords[j] = center[j] - radius + 2 * radius * prod
                z //= points
                dist += pow(coords[j] - center[j], 2)

            if dist <= radius_sq:
                val: int = np.linalg.norm(coords - server_estimate, ord=np.inf)

                if i == 0 or val > tmp_max:
                    tmp_max = val

                if i == 0 or val < tmp_min:
                    tmp_min = val

        return tmp_min, tmp_max
