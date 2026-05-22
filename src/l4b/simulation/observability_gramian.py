from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from l4b.simulation.analysis import SystemAnalysis
from l4b.simulation.controllability_gramian import GramianResult
from l4b.simulation.simulator import Simulator


class ObservabilityGramian(SystemAnalysis):
    """Finite-horizon observability Gramian.

    .. math::

        W_o(T) = \\sum_{t=0}^{T-1} (A^T)^t C^T C A^t

    A system is observable iff :math:`W_o(T)` is full rank for some finite
    :math:`T`. :attr:`result` is a :class:`GramianResult`.

    :param simulator: The simulator to analyse.
    :param T: Horizon. Defaults to ``simulator.state_dim``.
    :param tol: Rank tolerance passed to ``numpy.linalg.matrix_rank``.
    """

    def __init__(
        self,
        simulator: Simulator,
        T: int | None = None,
        tol: float = 1e-10,
    ) -> None:
        super().__init__(simulator)
        self._T: int = T if T is not None else simulator.state_dim
        self._tol = tol
        if self._T < 1:
            raise ValueError(f"T must be at least 1, got {self._T}")

    @property
    def result(self) -> GramianResult:
        return super().result  # type: ignore[return-value]

    def _compute(self) -> GramianResult:
        sim = self._simulator
        W: NDArray = np.zeros((sim.state_dim, sim.state_dim))
        At: NDArray = np.eye(sim.state_dim)

        for _ in range(self._T):
            CAt: NDArray = sim.C @ At
            W += CAt.T @ CAt
            At = sim.A @ At

        W = (W + W.T) / 2
        rank = int(np.linalg.matrix_rank(W, tol=self._tol))
        return GramianResult(
            W=W,
            rank=rank,
            is_full_rank=rank == sim.state_dim,
            condition=float(np.linalg.cond(W)),
        )
