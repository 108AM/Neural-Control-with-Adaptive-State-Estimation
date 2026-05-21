from __future__ import annotations

import numpy as np
from l4b.stats.analysis import Analysis
from l4b.dataset import Dataset
from numpy.typing import NDArray


class Correlation(Analysis):
    """Pearson correlation matrix between neurons.

    :param dataset: The dataset to analyse.
    :param trial: Trial index (0-based) to use. If ``None`` (default), data
        from all trials is concatenated along the time axis before computing
        the correlation.

    :attr:`result` has shape ``(n_neurons, n_neurons)``.
    """

    def __init__(self, dataset: Dataset, trial: int | None = None) -> None:
        super().__init__(dataset)
        self._trial = trial

    @property
    def result(self) -> NDArray:
        return super().result  # type: ignore[return-value]

    def _compute(self) -> NDArray:
        obs = self._dataset.observations  # (n_trials, n_timesteps, n_neurons)
        if self._trial is None:
            mat = obs.reshape(-1, self._dataset.n_neurons)
        else:
            mat = obs[self._trial]  # (n_timesteps, n_neurons)
        return np.corrcoef(mat.T)  # (n_neurons, n_neurons)
