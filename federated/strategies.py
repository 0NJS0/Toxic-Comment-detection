"""
Shared FL Metric Helpers
=========================

Only the metric-aggregation helper is kept here. The earlier FedProx /
LoRAAggregation strategy classes were removed:

- The actual FedAvg loop lives in ``federated/server.py`` and never
  called them (dead code).
- Their Flower-API signatures (`client_manager.sample(...)`) were tied to
  an obsolete Flower release and would not run against the pinned 1.13.

`weighted_average` remains a valid, reusable metric-aggregation callback
for Flower's evaluate strategy.
"""

from typing import List, Tuple
import numpy as np
from flwr.common import Metrics


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Compute a (num_examples-)weighted mean of scalar client metrics."""
    if not metrics:
        return {}
    num_total = sum(num for num, _ in metrics)
    if num_total == 0:
        return {}
    # Union of metric keys across clients.
    keys = {k for _, m in metrics for k in m}
    averages = {}
    for k in keys:
        weighted = sum(m.get(k, 0.0) * num for num, m in metrics)
        averages[k] = float(weighted / num_total)
    return averages