from __future__ import annotations

from typing import NamedTuple

import numpy as np
from analysis import Analysis
from dataset import Dataset
from numpy.typing import NDArray
from scipy.signal import welch


class PowerSpectrumResult(NamedTuple):
    frequencies: NDArray  # (n_freqs,)
    psd: NDArray  # (n_neurons, n_freqs)


class PowerSpectrum(Analysis):
    """Welch power spectral density per neuron, averaged across trials.

    :param dataset: The dataset to analyse.
    :param fs: Sampling frequency. Defaults to ``1.0`` (normalised units).
    :param nperseg: Welch segment length. Defaults to ``min(n_timesteps, 32)``.

    :attr:`result` is a :class:`PowerSpectrumResult`.
    """

    def __init__(
        self,
        dataset: Dataset,
        fs: float = 1.0,
        nperseg: int | None = None,
    ) -> None:
        super().__init__(dataset)
        self._fs = fs
        self._nperseg = nperseg if nperseg is not None else min(dataset.n_timesteps, 32)

    @property
    def result(self) -> PowerSpectrumResult:
        return super().result  # type: ignore[return-value]

    def _compute(self) -> PowerSpectrumResult:
        obs = self._dataset.observations  # (n_trials, n_timesteps, n_neurons)
        freqs, _ = welch(obs[0, :, 0], fs=self._fs, nperseg=self._nperseg)
        psd = np.zeros((self._dataset.n_neurons, len(freqs)))

        for n in range(self._dataset.n_neurons):
            trial_psds = []
            for t in range(self._dataset.n_trials):
                _, p = welch(obs[t, :, n], fs=self._fs, nperseg=self._nperseg)
                trial_psds.append(p)
            psd[n] = np.mean(trial_psds, axis=0)

        return PowerSpectrumResult(frequencies=freqs, psd=psd)
