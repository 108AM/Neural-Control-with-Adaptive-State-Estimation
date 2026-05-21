from __future__ import annotations

from typing import NamedTuple

import numpy as np
from l4b.stats.analysis import Analysis
from l4b.dataset import Dataset
from numpy.typing import NDArray
from scipy.signal import csd, welch


class CoherenceResult(NamedTuple):
    """Result of a :class:`Coherence` analysis.

    :param frequencies: Frequency bin centres in Hz (or normalised units if
        ``fs=1``), shape ``(n_freqs,)``. Produced by Welch's method; bins are
        linearly spaced from ``0`` to ``fs/2`` (Nyquist).
    :param coherence: Magnitude-squared coherence ``Cxy(f)`` for every ordered
        neuron pair, shape ``(n_neurons, n_neurons, n_freqs)``.
        ``coherence[i, j, :]`` is the coherence spectrum between neuron *i*
        (reference) and neuron *j* (target). The diagonal is identically 1.
        Values lie in ``[0, 1]``: 0 means no shared frequency content, 1 means
        the two signals are perfectly linearly related at that frequency.
    :param band_matrix: Mean coherence collapsed over a frequency band, shape
        ``(n_neurons, n_neurons)``. Computed by :meth:`Coherence.band_mean` and
        stored here so the illustrator can retrieve it without re-running the
        full spectral computation.
    """

    frequencies: NDArray  # (n_freqs,)
    coherence: NDArray  # (n_neurons, n_neurons, n_freqs)
    band_matrix: NDArray  # (n_neurons, n_neurons) — mean over the full band


class Coherence(Analysis):
    """Magnitude-squared coherence between every neuron pair, averaged across trials.

    For each pair (i, j) and each frequency bin f, coherence is:

        Cxy(f) = |Sxy(f)|² / (Sxx(f) · Syy(f))

    where Sxy(f) is the cross-spectral density between neurons i and j, and
    Sxx(f), Syy(f) are their respective auto-spectral densities.  All spectra
    are estimated via Welch's method.  Numerator and denominator are averaged
    separately across trials before dividing, which is more stable than
    averaging coherence values directly.

    The result also pre-computes :attr:`CoherenceResult.band_matrix` — the
    mean coherence collapsed over ``[f_low, f_high]`` — so that the illustrator
    can render a summary heatmap without re-running spectral estimation.

    :param dataset: The dataset to analyse.
    :param fs: Sampling frequency in Hz. Defaults to ``1.0`` (normalised
        units). Set this to the actual acquisition rate to get meaningful
        frequency axes.
    :param nperseg: Number of samples per Welch segment. Controls the
        frequency resolution (longer → finer bins) vs. variance trade-off.
        Defaults to ``min(n_timesteps, 32)``.
    :param f_low: Lower bound (inclusive) of the frequency band used to
        compute :attr:`CoherenceResult.band_matrix`. Defaults to ``0.0``.
    :param f_high: Upper bound (inclusive) of the frequency band used to
        compute :attr:`CoherenceResult.band_matrix`. Defaults to ``None``
        (Nyquist — the full spectrum).

    :attr:`result` is a :class:`CoherenceResult`.
    """

    def __init__(
        self,
        dataset: Dataset,
        fs: float = 1.0,
        nperseg: int | None = None,
        f_low: float = 0.0,
        f_high: float | None = None,
    ) -> None:
        super().__init__(dataset)
        self._fs = fs
        self._nperseg = nperseg if nperseg is not None else min(dataset.n_timesteps, 32)
        self._f_low = f_low
        self._f_high = f_high

    @property
    def result(self) -> CoherenceResult:
        return super().result  # type: ignore[return-value]

    def band_mean(self, frequencies: NDArray, coherence: NDArray) -> NDArray:
        """Collapse ``coherence`` to a ``(n_neurons, n_neurons)`` matrix by averaging over a band.

        Selects frequency bins where ``f_low <= f <= f_high`` (as supplied at
        construction time) and returns the mean coherence across those bins.
        This is stored on :attr:`CoherenceResult.band_matrix` so callers never
        need to repeat the computation.

        :param frequencies: The ``(n_freqs,)`` frequency array from Welch.
        :param coherence: The ``(n_neurons, n_neurons, n_freqs)`` coherence array.
        :returns: ``(n_neurons, n_neurons)`` mean magnitude-squared coherence
            over the requested band.
        """
        f_high = frequencies[-1] if self._f_high is None else self._f_high
        mask = (frequencies >= self._f_low) & (frequencies <= f_high)
        return coherence[:, :, mask].mean(axis=2)

    def _compute(self) -> CoherenceResult:
        obs = self._dataset.observations  # (n_trials, n_timesteps, n_neurons)
        n_neurons = self._dataset.n_neurons
        n_trials = self._dataset.n_trials

        freqs, _ = welch(obs[0, :, 0], fs=self._fs, nperseg=self._nperseg)
        n_freqs = len(freqs)

        coherence = np.zeros((n_neurons, n_neurons, n_freqs))

        for i in range(n_neurons):
            for j in range(n_neurons):
                if i == j:
                    coherence[i, j, :] = 1.0
                    continue

                cxy_trials = []
                sxx_trials = []
                syy_trials = []

                for t in range(n_trials):
                    x = obs[t, :, i]
                    y = obs[t, :, j]
                    _, pxx = welch(x, fs=self._fs, nperseg=self._nperseg)
                    _, pyy = welch(y, fs=self._fs, nperseg=self._nperseg)
                    _, sxy = csd(x, y, fs=self._fs, nperseg=self._nperseg)
                    sxx_trials.append(pxx)
                    syy_trials.append(pyy)
                    cxy_trials.append(sxy)

                sxx = np.mean(sxx_trials, axis=0)
                syy = np.mean(syy_trials, axis=0)
                sxy = np.mean(cxy_trials, axis=0)

                denom = sxx * syy
                coherence[i, j] = np.where(denom > 0, np.abs(sxy) ** 2 / denom, 0.0)

        return CoherenceResult(
            frequencies=freqs,
            coherence=coherence,
            band_matrix=self.band_mean(freqs, coherence),
        )
