import torch
from sphere_exploration.max_min_on_sphere import ParameterSpaceExplorer


class TorchMonteCarlo(ParameterSpaceExplorer):
    def compute(
        self,
        center: torch.Tensor,
        server_estimate: torch.Tensor,
        radius: int,
        num_samples: int = 100_000,
        batch_size: int = 1000,
    ) -> tuple[float, float]:
        """
        We can use Monte Carlo sampling to approximate the min and max L-infinity norm values.
        By using Monte Carlo sampling, we avoid the need to create a large tensor containing all possible combinations
            of points, and we reduce the computational complexity of the problem.
        You can control the trade-off between accuracy and performance by adjusting the num_samples parameter.
        Increasing the number of samples will provide a more accurate approximation,
            but it will also require more computation time.
        Note that this implementation provides an approximate solution rather than an exact one.
        However, in high-dimensional problems like this, the approximate solution is often sufficient and
        much more computationally feasible.


        :param center:
        :param server_estimate:
        :param radius:
        :param num_samples:
        :return:
        """
        dim_size: int = center.size(0)
        device: str = center.device

        # Initialize variables to store min and max L-infinity norm
        min_val: float = float("inf")
        max_val: float = float("-inf")

        num_batches = (num_samples + batch_size - 1) // batch_size
        for _ in range(num_batches):
            # Generate random samples uniformly within a hypercube with side length 2*radius centered at the origin
            batch_samples: torch.Tensor = torch.rand((batch_size, dim_size), device=device) * 2 * radius - radius

            # Check if the samples are within the specified sphere
            squared_dists: torch.Tensor = torch.sum(batch_samples ** 2, dim=1)
            valid_samples = batch_samples[squared_dists <= radius ** 2]

            if valid_samples.size(0) > 0:
                # Shift the valid samples by the center
                valid_samples += center

                # Calculate the L-infinity norm between valid_samples and server_estimate
                vals = torch.norm(valid_samples - server_estimate, p=float("inf"), dim=1)

                # Update min and max values
                min_val = min(min_val, torch.min(vals).item())
                max_val = max(max_val, torch.max(vals).item())

        return min_val, max_val
