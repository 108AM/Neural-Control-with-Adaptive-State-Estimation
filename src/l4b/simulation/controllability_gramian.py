from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from l4b.simulation.analysis import SystemAnalysis
from l4b.simulation.simulator import Simulator


class GramianResult(NamedTuple):
    W: NDArray  # (state_dim, state_dim)
    rank: int
    is_full_rank: bool
    condition: float


class ControllabilityGramian(SystemAnalysis):
    """Finite-horizon controllability Gramian.

    .. math::

        W_c(T) = \\sum_{t=0}^{T-1} A^t B B^T (A^T)^t

    A system is controllable iff :math:`W_c(T)` is full rank for some finite
    :math:`T`. :attr:`result` is a :class:`GramianResult`.

    :param simulator: The simulator to analyse.
    :param T: Horizon. Defaults to ``simulator.state_dim``, which is the
        minimum required for controllability to be detectable.
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
            AtB: NDArray = At @ sim.B
            W += AtB @ AtB.T
            At = At @ sim.A

        W = (W + W.T) / 2
        rank = int(np.linalg.matrix_rank(W, tol=self._tol))
        return GramianResult(
            W=W,
            rank=rank,
            is_full_rank=rank == sim.state_dim,
            condition=float(np.linalg.cond(W)),
        )
