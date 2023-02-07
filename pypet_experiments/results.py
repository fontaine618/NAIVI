import pandas as pd
from typing import Any


class Results:

    def __init__(
            self,
            training_metrics: dict[str, float],
            testing_metrics: dict[str, float],
            estimation_metrics: dict[str, float],
            logs: Any = None
    ):
        self._metrics = {
            "training": training_metrics,
            "testing": testing_metrics,
            "estimation": estimation_metrics
        }
        self._logs = logs

    def to_dict(self) -> dict[str, float]:
        out = dict()
        for key, value in self._metrics.items():
            for metric, val in value.items():
                out[f"{key}.{metric}"] = val
        for key, value in self._logs.items():
            out[f"logs.{key}"] = value
        return out
