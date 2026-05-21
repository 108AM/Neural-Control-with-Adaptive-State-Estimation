from __future__ import annotations

from typing import NamedTuple

import numpy as np
from l4b.stats.analysis import Analysis
from l4b.dataset import Dataset
from numpy.typing import NDArray


class CrossCorrelationResult(NamedTuple):
    ccf: NDArray  # (2 * max_lag + 1,)
    lags: NDArray  # (2 * max_lag + 1,) integer lag indices


class CrossCorrelation(Analysis):
    """Normalized cross-correlation between two neurons, averaged across trials.

    :param dataset: The dataset to analyse.
    :param neuron_a: Index of the reference neuron.
    :param neuron_b: Index of the target neuron.
    :param max_lag: Maximum lag to compute (inclusive). Defaults to
        ``n_timesteps // 2``.

    :attr:`result` is a :class:`CrossCorrelationResult`.
    """

    def __init__(
        self,
        dataset: Dataset,
        neuron_a: int = 0,
        neuron_b: int = 1,
        max_lag: int | None = None,
    ) -> None:
        super().__init__(dataset)
        self._neuron_a = neuron_a
        self._neuron_b = neuron_b
        self._max_lag = max_lag if max_lag is not None else dataset.n_timesteps // 2

    @property
    def result(self) -> CrossCorrelationResult:
        return super().result  # type: ignore[return-value]

    def _compute(self) -> CrossCorrelationResult:
        obs = self._dataset.observations  # (n_trials, n_timesteps, n_neurons)
        lags = np.arange(-self._max_lag, self._max_lag + 1)
        ccf_trials = []

        for t in range(self._dataset.n_trials):
            x = obs[t, :, self._neuron_a]
            y = obs[t, :, self._neuron_b]
            x = x - x.mean()
            y = y - y.mean()
            full = np.correlate(x, y, mode="full")  # length 2*n_timesteps - 1
            centre = len(full) // 2
            ccf_slice = full[centre - self._max_lag : centre + self._max_lag + 1]
            norm = np.sqrt(np.dot(x, x) * np.dot(y, y))
            ccf_trials.append(
                ccf_slice / norm if norm > 0 else np.zeros_like(ccf_slice)
            )

        return CrossCorrelationResult(
            ccf=np.mean(ccf_trials, axis=0),
            lags=lags,
        )
