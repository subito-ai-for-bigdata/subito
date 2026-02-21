import torch


class GMUtils:
    """Utilities for Geometric Monitoring actions."""

    @staticmethod
    def _fft(vector: torch.Tensor, sample_size: int) -> torch.Tensor:
        vector = vector.to('cuda')
        vector_coeff: torch.Tensor = torch.fft.fft(vector, norm='ortho')
        abs_vec: torch.Tensor = torch.abs(vector_coeff)
        top_coeffs: torch.Tensor = torch.topk(abs_vec, k=sample_size, largest=True, sorted=True).values
        top_coeffs = top_coeffs.to('cpu')
        return top_coeffs[:sample_size]

    @staticmethod
    def get_tensors_from_model_grads(worker_grads: list[torch.Tensor]) -> list[torch.Tensor]:
        t_view: list[torch.Tensor] = [tensor.view(-1) for tensor in worker_grads]
        return t_view

    @staticmethod
    def compute_local_vector(worker_grads: list[torch.Tensor], fft_sample: int = 0) -> torch.Tensor:
        grad_tensors: list[torch.Tensor] = GMUtils.get_tensors_from_model_grads(worker_grads)
        local_vi: torch.Tensor = torch.cat(grad_tensors, dim=0)
        if fft_sample > 0:
            local_vi = GMUtils._fft(local_vi, fft_sample)
        return local_vi

    @staticmethod
    def compute_delta_local_vectors(cur_worker_vector: torch.Tensor, prev_worker_vector: torch.Tensor) -> torch.Tensor:
        """"""
        delta_vi: torch.Tensor = cur_worker_vector - prev_worker_vector
        return delta_vi

    @staticmethod
    def compute_estimate(v_i_workers: dict[str, torch.Tensor], weights_workers: dict[str, float]) -> torch.Tensor:
        """
        Weights are expected to sum up to 1 and be floats.

        :param v_i_workers:
        :param weights_workers:
        :return:
        """

        weighted_vectors: list = []
        for w_name, w_grads in v_i_workers.items():
            w_weight: float = weights_workers[w_name]
            scaled_v_i: torch.Tensor = torch.mul(w_weight, w_grads)
            weighted_vectors.append(scaled_v_i)

        estimate: torch.Tensor = torch.sum(torch.stack(weighted_vectors), dim=0)
        return estimate

    @staticmethod
    def compute_center(average_vector_at_last_sync: torch.Tensor, u_i: torch.Tensor, lr: float) -> torch.Tensor:
        """

        :param average_vector_at_last_sync:
        :param u_i:
        :return:
        """
        new_vec = average_vector_at_last_sync + u_i*(1/2)
        scaled_vector: torch.Tensor = new_vec*(1/2)
        return new_vec

    @staticmethod
    def compute_radius(average_vector_at_last_sync: torch.Tensor, u_i: torch.Tensor, lr: float, vector_distance: callable) -> torch.Tensor:
        """

        :param average_vector_at_last_sync:
        :param u_i:
        :param vector_distance:
        :return:
        """
        radius: torch.Tensor = u_i
        radius = vector_distance(radius)
        radius = radius*(1/2)
        return radius
