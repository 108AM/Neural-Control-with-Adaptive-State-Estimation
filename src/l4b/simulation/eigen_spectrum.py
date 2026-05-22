from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from l4b.simulation.analysis import SystemAnalysis
from l4b.simulation.simulator import Simulator


class EigenSpectrumResult(NamedTuple):
    eigenvalues: NDArray  # complex, shape (n,)
    spectral_radius: float  # max |eigenvalue|
    is_schur_stable: bool  # spectral_radius < tol


class EigenSpectrum(SystemAnalysis):
    """Eigen-spectrum of the dynamics matrix :math:`A`.

    Computes the eigenvalues, spectral radius
    :math:`\\rho(A) = \\max_i |\\lambda_i(A)|`, and a Schur-stability flag
    (:math:`\\rho(A) < \\text{tol}`). For a discrete-time system, Schur
    stability with ``tol=1`` is the standard stability criterion.
    :attr:`result` is an :class:`EigenSpectrumResult`.

    Named ``EigenSpectrum`` (rather than the more common ``Spectrum``) to
    avoid confusion with :class:`l4b.stats.power_spectrum.PowerSpectrum`,
    which computes a frequency-domain spectrum of observations.

    :param simulator: The simulator whose dynamics matrix is to be analysed.
    :param tol: Threshold against which :math:`\\rho` is compared for the
        ``is_schur_stable`` flag. Defaults to ``1.0`` (the discrete-time
        stability boundary).
    """

    def __init__(self, simulator: Simulator, tol: float = 1.0) -> None:
        super().__init__(simulator)
        self._tol = tol

    @property
    def result(self) -> EigenSpectrumResult:
        return super().result  # type: ignore[return-value]

    def _compute(self) -> EigenSpectrumResult:
        eigenvalues: NDArray = np.linalg.eigvals(self._simulator.A)
        spectral_radius = float(np.max(np.abs(eigenvalues)))
        return EigenSpectrumResult(
            eigenvalues=eigenvalues,
            spectral_radius=spectral_radius,
            is_schur_stable=spectral_radius < self._tol,
        )
