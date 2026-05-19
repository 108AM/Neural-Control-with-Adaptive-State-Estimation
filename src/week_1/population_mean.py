from __future__ import annotations

from analysis import Analysis
from numpy.typing import NDArray


class PopulationMean(Analysis):
    """Mean neural activity averaged across trials.

    :attr:`result` has shape ``(n_timesteps, n_neurons)``.
    """

    def _compute(self) -> NDArray:
        return self._dataset.observations.mean(axis=0)

    @property
    def result(self) -> NDArray:
        return super().result  # type: ignore[return-value]
