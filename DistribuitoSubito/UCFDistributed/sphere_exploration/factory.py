from sphere_exploration.max_min_on_sphere import ParameterSpaceExplorer
from sphere_exploration.np_vectorized import NpVectorized
from sphere_exploration.standard import Standard
from sphere_exploration.torch_annealing import TorchAnnealing
from sphere_exploration.torch_monte_carlo import TorchMonteCarlo
from sphere_exploration.torch_standard import TorchStandard


class SphereExplorationFactory:
    """Factory class for SphereExploration objects"""

    @staticmethod
    def create(imp: str, *args, **kwargs) -> ParameterSpaceExplorer:
        if imp == "standard":
            return Standard(*args, **kwargs)
        elif imp == "np_vectorized":
            return NpVectorized(*args, **kwargs)
        elif imp == "torch_standard":
            return TorchStandard(*args, **kwargs)
        elif imp == "torch_annealing":
            return TorchAnnealing(*args, **kwargs)
        elif imp == "torch_monte_carlo":
            return TorchMonteCarlo(*args, **kwargs)
        else:
            raise ValueError(f"Unknown annealing implementation: {imp}")
