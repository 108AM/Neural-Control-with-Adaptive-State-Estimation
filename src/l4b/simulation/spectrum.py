from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from l4b.simulation.analysis import MatrixAnalysis


class SpectrumResult(NamedTuple):
    eigenvalues: NDArray  # complex, shape (n,)
    spectral_radius: float  # max |eigenvalue|
    is_schur_stable: bool  # spectral_radius < tol


class Spectrum(MatrixAnalysis):
    """Eigen-spectrum of a square matrix.

    Computes the eigenvalues, spectral radius
    :math:`\\rho(A) = \\max_i |\\lambda_i(A)|`, and a Schur-stability flag
    (:math:`\\rho(A) < \\text{tol}`). For a discrete-time dynamics matrix
    ``A``, Schur stability with ``tol=1`` is the standard stability
    criterion. :attr:`result` is a :class:`SpectrumResult`.

    :param matrix: The square matrix to analyse.
    :param tol: Threshold against which :math:`\\rho` is compared for the
        ``is_schur_stable`` flag. Defaults to ``1.0`` (the discrete-time
        stability boundary).
    """

    def __init__(self, matrix: NDArray, tol: float = 1.0) -> None:
        super().__init__(matrix)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"Spectrum requires a square matrix, got shape {matrix.shape}"
            )
        self._tol = tol

    @property
    def result(self) -> SpectrumResult:
        return super().result  # type: ignore[return-value]

    def _compute(self) -> SpectrumResult:
        eigenvalues: NDArray = np.linalg.eigvals(self._matrix)
        spectral_radius = float(np.max(np.abs(eigenvalues)))
        return SpectrumResult(
            eigenvalues=eigenvalues,
            spectral_radius=spectral_radius,
            is_schur_stable=spectral_radius < self._tol,
        )
