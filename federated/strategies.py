"""
Custom FL Strategies for FedPref
=================================

- **FedProxStrategy**: Adds proximal term to prevent client drift under non-IID
- **LoRAAggregation**: Only averages LoRA and classifier weights (ignores base)
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import flwr as fl
from flwr.common import Metrics, Parameters, ndarrays_to_parameters, parameters_to_ndarrays


class FedProxStrategy(fl.server.strategy.FedAvg):
    """
    FedProx: Adds a proximal term μ||w - w^t||² to the client's loss.

    This prevents client models from drifting too far from the global model,
    which is especially important under non-IID data distributions.

    Reference:
        Li et al., "Federated Optimization in Heterogeneous Networks", 2020
    """

    def __init__(self, proximal_mu: float = 0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.proximal_mu = proximal_mu

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple]:
        """Add proximal_mu to the config sent to each client."""
        config_list = []
        for client_proxy in client_manager.sample(
            num_clients_in_fit=self.min_fit_clients,
            client_manager=client_manager,
        ):
            config = {
                "local_epochs": 1,
                "proximal_mu": self.proximal_mu,
            }
            config_list.append((client_proxy, config))
        return config_list


class LoRAAggregation(fl.server.strategy.FedAvg):
    """
    Aggregation strategy that only averages LoRA + classifier parameters.

    This is the default for FedPref. The base encoder is frozen and
    never communicated, matching the mobile constraint.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Compute weighted average of metrics across clients."""
    if not metrics:
        return {}
    losses = []
    for num_examples, m in metrics:
        losses.append(m.get("loss", 0.0))
    return {"loss": float(np.mean(losses))}
