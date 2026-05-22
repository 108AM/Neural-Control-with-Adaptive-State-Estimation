from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any

from numpy.typing import NDArray

from l4b.simulation.simulator import Simulator


class SystemAnalysis(ABC):
    """Abstract base for lazy analyses of a :class:`Simulator`'s system.

    Direct analogue of :class:`l4b.stats.analysis.Analysis` (parameterised by
    a :class:`l4b.dataset.Dataset`) and sibling of :class:`MatrixAnalysis`
    (parameterised by a single matrix). Subclasses implement
    :meth:`_compute`; the result is computed on first access of
    :attr:`result` and cached thereafter via ``functools.cached_property``.

    :param simulator: The simulator whose system properties are to be analysed.
    """

    def __init__(self, simulator: Simulator) -> None:
        self._simulator = simulator

    @cached_property
    def result(self) -> Any:
        return self._compute()

    @abstractmethod
    def _compute(self) -> Any: ...


class MatrixAnalysis(ABC):
    """Abstract base for lazy analyses of a single matrix.

    Matrix-parameterised analogue of :class:`SystemAnalysis` and
    :class:`l4b.stats.analysis.Analysis`. Use this when the analysis depends
    only on one matrix (e.g. the dynamics matrix ``A``) rather than on a
    full :class:`Simulator` or :class:`l4b.dataset.Dataset`. Subclasses
    implement :meth:`_compute`; :attr:`result` is cached on first access.

    :param matrix: The matrix to analyse.
    """

    def __init__(self, matrix: NDArray) -> None:
        self._matrix = matrix

    @cached_property
    def result(self) -> Any:
        return self._compute()

    @abstractmethod
    def _compute(self) -> Any: ...
