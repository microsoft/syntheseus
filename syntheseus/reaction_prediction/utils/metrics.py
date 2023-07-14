from dataclasses import dataclass
from typing import List, TypeVar

import numpy as np

OutputType = TypeVar("OutputType")


class TopKMetricsAccumulator:
    """Class to accumulate prediction top-k accuracy and MRR under a given notion of correctness."""

    def __init__(self, max_num_results: int):
        self._max_num_results = max_num_results

        # Initialize things we will need to compute the metrics.
        self._top_k_correct_cnt = np.zeros(max_num_results)
        self._sum_reciprocal_rank = 0.0
        self._num_samples = 0

    def add(self, is_output_correct: List[bool]) -> None:
        assert len(is_output_correct) <= self._max_num_results

        self._num_samples += 1
        for idx, output in enumerate(is_output_correct):
            if output:
                self._top_k_correct_cnt[idx:] += 1.0
                self._sum_reciprocal_rank += 1.0 / (idx + 1)
                break

    @property
    def num_samples(self) -> int:
        return self._num_samples

    @property
    def top_k(self) -> List[float]:
        return list(self._top_k_correct_cnt / self._num_samples)

    @property
    def mrr(self) -> float:
        return self._sum_reciprocal_rank / self._num_samples


@dataclass(frozen=True)
class ModelTimingResults:
    time_model_call: float
    time_post_processing: float


def compute_total_time(timing_results: List[ModelTimingResults]) -> ModelTimingResults:
    return ModelTimingResults(
        **{
            key: sum(getattr(result, key) for result in timing_results)
            for key in ["time_model_call", "time_post_processing"]
        }
    )
