from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from l4b.simulation.analysis import SystemAnalysis
from l4b.simulation.simulator import Simulator


class EigenpairsResult(NamedTuple):
    eigenvalues: NDArray  # complex, shape (n,)
    eigenvectors: NDArray  # complex, shape (n, n); column k is the right
    # eigenvector for ``eigenvalues[k]``


class Eigenpairs(SystemAnalysis):
    """Right eigenpairs :math:`(\\lambda_k, v_k)` of the dynamics matrix
    :math:`A`, where :math:`A v_k = \\lambda_k v_k`.

    Complements :class:`l4b.simulation.eigen_spectrum.EigenSpectrum`: where
    :class:`EigenSpectrum` summarises the spectrum (radius, stability),
    :class:`Eigenpairs` also exposes the eigenvectors needed for modal
    analysis — projecting trajectories onto eigenmodes, inspecting the
    conditioning :math:`\\kappa(V)` of the eigenbasis, or constructing the
    diagonalisation :math:`A = V \\Lambda V^{-1}`.

    :attr:`result` is an :class:`EigenpairsResult`.

    :param simulator: The simulator whose dynamics matrix is to be analysed.
    """

    def __init__(self, simulator: Simulator) -> None:
        super().__init__(simulator)

    @property
    def result(self) -> EigenpairsResult:
        return super().result  # type: ignore[return-value]

    def _compute(self) -> EigenpairsResult:
        eigenvalues, eigenvectors = np.linalg.eig(self._simulator.A)
        return EigenpairsResult(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
        )
