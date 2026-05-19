from __future__ import annotations

from typing import NamedTuple

import numpy as np
from analysis import Analysis
from dataset import Dataset
from numpy.typing import NDArray


class MeanResult(NamedTuple):
    mean: NDArray  # (n_timesteps, n_neurons)
    sem: NDArray  # (n_timesteps, n_neurons)


class Mean(Analysis):
    """Trial-averaged activity and standard error of the mean.

    :attr:`result` is a :class:`MeanResult` with ``mean`` and ``sem`` fields,
    both of shape ``(n_timesteps, n_neurons)``.
    """

    def __init__(self, dataset: Dataset) -> None:
        super().__init__(dataset)

    @property
    def result(self) -> MeanResult:
        return super().result  # type: ignore[return-value]

    def _compute(self) -> MeanResult:
        obs = self._dataset.observations  # (n_trials, n_timesteps, n_neurons)
        mean = obs.mean(axis=0)
        sem = obs.std(axis=0, ddof=1) / np.sqrt(self._dataset.n_trials)
        return MeanResult(mean=mean, sem=sem)
