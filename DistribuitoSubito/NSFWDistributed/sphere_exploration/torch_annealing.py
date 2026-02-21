from functools import partial

import torch
from sphere_exploration.max_min_on_sphere import ParameterSpaceExplorer


class TorchAnnealing(ParameterSpaceExplorer):
    def random_point_in_sphere(
        self, dim_size: int, radius: torch.Tensor, radius_sq: torch.Tensor, center: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        while True:
            point: torch.Tensor = torch.rand(dim_size, device=device) * 2 * radius - radius
            if torch.sum(torch.pow(point, 2)) <= radius_sq:
                return point + center

    def acceptance_probability(self, delta: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        return torch.exp(-delta / T)

    def compute(
        self,
        center: torch.Tensor,
        server_estimate: torch.Tensor,
        radius: int,
        num_iterations: int = 1000,
        T_init: float = 10_000,
        T_final: float = 0,
        cooling_rate: float = 0.999,
    ) -> tuple[float, float]:
        """
        This implementation uses Simulated Annealing to search for the min and max L-infinity norms. You can control
        the number of iterations and the temperature schedule using the num_iterations, T_init, T_final,
        and cooling_rate parameters.

        Do note that Simulated Annealing provides an approximate solution and its performance depends on the quality
        of the temperature schedule and the number of iterations. Experimenting with these parameters can help you
        find a balance between accuracy and computational cost.
        """
        dim_size: int = center.size(0)
        device: str = center.device
        radius_sq: torch.Tensor = torch.pow(radius, 2)

        # Initialize temperature, min_val, and max_val
        T: torch.Tensor = torch.tensor(T_init, device=device)
        T_final: torch.Tensor = torch.tensor(T_final, device=device)
        min_val: torch.Tensor = torch.tensor(float("inf"), device=device)
        max_val: torch.Tensor = torch.tensor(float("-inf"), device=device)

        # Partial inits
        random_point_gen: callable = partial(
            self.random_point_in_sphere,
            center=center,
            radius=radius,
            radius_sq=radius_sq,
            device=device,
            dim_size=dim_size,
        )
        acceptance_gen: callable = partial(self.acceptance_probability, T=T)

        # Initialize current solution
        current_solution: torch.Tensor = random_point_gen()
        current_val: torch.Tensor = torch.norm(current_solution - server_estimate, p=float("inf"))

        for _ in range(num_iterations):
            # Generate a random neighbor of the current solution
            neighbor = random_point_gen()
            neighbor_val = torch.norm(neighbor - server_estimate, p=float("inf"))

            # Calculate the difference in objective function values
            delta = neighbor_val - current_val

            # Determine if the neighbor should be accepted as the new solution
            if delta < 0 or acceptance_gen(delta) > torch.rand(1, device=device):
                current_solution: torch.Tensor = neighbor
                current_val: torch.Tensor = neighbor_val

                # Update min and max values
                min_val: torch.Tensor = torch.min(min_val, current_val)
                max_val: torch.Tensor = torch.max(max_val, current_val)

            # Update the temperature
            T *= cooling_rate
            if T <= T_final:
                break

        _min, _max = min_val.item(), max_val.item()

        if _min == float("-inf") or _max == float("inf"):
            self.logger.debug("Produced inf/-inf results.")
            return 0, 1
