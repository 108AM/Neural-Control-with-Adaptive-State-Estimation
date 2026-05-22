from __future__ import annotations

import numpy as np
from l4b.stats.analysis import Analysis
from l4b.dataset import Dataset
from numpy.typing import NDArray


class Correlation(Analysis):
    """Pearson correlation matrix between neurons with an optional time lag.

    Entry ``[i, j]`` of :attr:`result` is the Pearson correlation between the
    time series of neuron *i* at time *t* and the time series of neuron *j* at
    time *t + time_lag*.  At ``time_lag=1`` (the default) this is a one-step
    predictive correlation; at higher lags longer-range temporal dependencies
    are captured.  Because the two slices are generally different, the result
    matrix is **not** guaranteed to be symmetric.

    :param dataset: The dataset to analyse.
    :param trial: Trial index (0-based) to use. If ``None`` (default), data
        from all trials is concatenated along the time axis before computing
        the correlation.
    :param time_lag: Number of timesteps by which the *j*-axis observations
        are shifted forward relative to the *i*-axis observations.  Must be
        geq 0.  Defaults to ``0`` (no lag).

    :attr:`result` has shape ``(n_neurons, n_neurons)``.
    """

    def __init__(
        self,
        dataset: Dataset,
        trial: int | None = None,
        time_lag: int = 0,
    ) -> None:
        if time_lag < 0:
            raise ValueError(
                f"time_lag must be geq 0, got {time_lag!r}."
            )
        super().__init__(dataset)
        self._trial = trial
        self._time_lag = time_lag

    @property
    def result(self) -> NDArray:
        return super().result  # type: ignore[return-value]

    def _compute(self) -> NDArray:
        obs = self._dataset.observations
        lag = self._time_lag
        if self._trial is None:
            mat = obs.reshape(-1, self._dataset.n_neurons)
        else:
            mat = obs[self._trial]

        mat_past = mat[:mat.shape[0] - lag] # Important to include mat.shape[0] term as lag may be 0
        mat_future = mat[lag:]

        n = self._dataset.n_neurons
        full = np.corrcoef(mat_past.T, mat_future.T)
        # Top-right nxn block is the correlation matrix.
        return full[:n, n:]
