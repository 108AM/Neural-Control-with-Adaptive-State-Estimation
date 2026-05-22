from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any

from l4b.simulation.simulator import Simulator


class SystemAnalysis(ABC):
    """Abstract base for lazy analyses of a :class:`Simulator`'s system.

    Direct analogue of :class:`l4b.stats.analysis.Analysis` (parameterised by
    a :class:`l4b.dataset.Dataset`). Subclasses implement :meth:`_compute`;
    the result is computed on first access of :attr:`result` and cached
    thereafter via ``functools.cached_property``.

    :param simulator: The simulator whose system properties are to be analysed.
    """

    def __init__(self, simulator: Simulator) -> None:
        self._simulator = simulator

    @cached_property
    def result(self) -> Any:
        return self._compute()

    @abstractmethod
    def _compute(self) -> Any: ...
