from __future__ import annotations

from typing import NamedTuple

import numpy as np
from l4b.stats.analysis import Analysis
from l4b.dataset import Dataset
from numpy.typing import NDArray


class AutocorrelationResult(NamedTuple):
    acf: NDArray  # (n_neurons, max_lag + 1)
    lags: NDArray  # (max_lag + 1,) integer lag indices


class Autocorrelation(Analysis):
    """Per-neuron autocorrelation function averaged across trials.

    :param dataset: The dataset to analyse.
    :param max_lag: Maximum lag to compute (inclusive). Defaults to
        ``n_timesteps // 2``.

    :attr:`result` is an :class:`AutocorrelationResult`.
    """

    def __init__(self, dataset: Dataset, max_lag: int | None = None) -> None:
        super().__init__(dataset)
        self._max_lag = max_lag if max_lag is not None else dataset.n_timesteps // 2

    @property
    def result(self) -> AutocorrelationResult:
        return super().result  # type: ignore[return-value]

    def _compute(self) -> AutocorrelationResult:
        obs = self._dataset.observations  # (n_trials, n_timesteps, n_neurons)
        lags = np.arange(self._max_lag + 1)
        acf = np.zeros((self._dataset.n_neurons, self._max_lag + 1))

        for n in range(self._dataset.n_neurons):
            trial_acfs = []
            for t in range(self._dataset.n_trials):
                x = obs[t, :, n]
                x = x - x.mean()
                full = np.correlate(x, x, mode="full")
                one_sided = full[len(full) // 2 :]  # lags 0 .. n_timesteps-1
                norm = one_sided[0]
                if norm == 0:
                    trial_acfs.append(np.zeros(self._max_lag + 1))
                else:
                    trial_acfs.append(one_sided[: self._max_lag + 1] / norm)
            acf[n] = np.mean(trial_acfs, axis=0)

        return AutocorrelationResult(acf=acf, lags=lags)
