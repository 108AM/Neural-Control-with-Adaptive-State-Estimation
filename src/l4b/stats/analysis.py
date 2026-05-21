from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any

from l4b.dataset import Dataset


class Analysis(ABC):
    """Abstract base for lazy dataset analyses.

    Subclasses implement :meth:`_compute`. The result is computed on first
    access of :attr:`result` and cached for all subsequent accesses via
    ``functools.cached_property``.

    :param dataset: The dataset to analyse.
    """

    def __init__(self, dataset: Dataset) -> None:
        self._dataset = dataset

    @cached_property
    def result(self) -> Any:
        return self._compute()

    @abstractmethod
    def _compute(self) -> Any: ...
