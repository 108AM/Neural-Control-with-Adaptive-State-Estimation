from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class Dataset:
    """Immutable container for a neural recording dataset.

    :param observations: Array of shape ``(n_trials, n_timesteps, n_neurons)``.
    """

    _observations: NDArray = field(repr=False)

    def __init__(self, observations: ArrayLike) -> None:
        arr = np.asarray(observations, dtype=float)
        if arr.ndim != 3:
            raise ValueError(
                f"observations must be 3-D (n_trials, n_timesteps, n_neurons); "
                f"got {arr.ndim}-D."
            )
        object.__setattr__(self, "_observations", arr)

    @property
    def observations(self) -> NDArray:
        return self._observations

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._observations.shape  # type: ignore[return-value]

    @property
    def n_trials(self) -> int:
        return self._observations.shape[0]

    @property
    def n_timesteps(self) -> int:
        return self._observations.shape[1]

    @property
    def n_neurons(self) -> int:
        return self._observations.shape[2]

    def select_trials(
        self, indices: Union[NDArray, list[int], slice]
    ) -> "Dataset":
        """Return a new Dataset containing only the trials at *indices*.

        Fancy indexing copies the data, so the original dataset is not mutated.
        """
        return Dataset(self._observations[indices])

    def __repr__(self) -> str:
        return (
            f"Dataset(n_trials={self.n_trials}, n_timesteps={self.n_timesteps}, "
            f"n_neurons={self.n_neurons})"
        )
