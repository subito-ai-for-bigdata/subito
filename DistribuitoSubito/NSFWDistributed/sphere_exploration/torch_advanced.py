import torch
from sphere_exploration.max_min_on_sphere import ParameterSpaceExplorer


class TorchAdvanced(ParameterSpaceExplorer):
    def compute(
        self, center: torch.Tensor, server_estimate: torch.Tensor, radius: int, points: int = 6
    ) -> tuple[float, float]:
        raise NotImplementedError()
