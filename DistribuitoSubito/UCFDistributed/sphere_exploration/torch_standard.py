import torch
from sphere_exploration.max_min_on_sphere import ParameterSpaceExplorer


class TorchStandard(ParameterSpaceExplorer):
    def compute(
        self, center: torch.Tensor, server_estimate: torch.Tensor, radius: int, points: int = 6
    ) -> tuple[float, float]:
        dim_size = center.size(0)
        radius_sq = radius ** 2

        # Create a tensor with all possible combinations of points
        linspaces: list[torch.Tensor] = [torch.linspace(c - radius, c + radius, points) for c in center]
        ranges: tuple[torch.Tensor] = torch.stack(linspaces).unbind()
        meshgrid: tuple[torch.Tensor] = torch.meshgrid(*ranges, indexing="ij")
        grid: torch.Tensor = torch.stack(meshgrid)
        coords: torch.Tensor = grid.reshape(dim_size, -1).T

        # Calculate the squared distances from the center
        squared_dists: torch.Tensor = torch.sum((coords - center) ** 2, dim=1)

        # Get the valid coordinates with squared distances <= radius_sq
        valid_coords: torch.Tensor = coords[squared_dists <= radius_sq]

        # Calculate the L-infinity norm between valid_coords and server_estimate
        vals: torch.Tensor = torch.norm(valid_coords - server_estimate, p=float("inf"), dim=1)

        # Get min and max values
        tmp_min: float = torch.min(vals).item()
        tmp_max: float = torch.max(vals).item()

        return tmp_min, tmp_max
