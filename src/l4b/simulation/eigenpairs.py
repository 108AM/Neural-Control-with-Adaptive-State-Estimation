import numpy as np

from numpy.typing import NDArray
from typing import Any

from l4b.simulation import MatrixAnalysis


class Eigenpairs(MatrixAnalysis):
    def __init__(self, matrix: NDArray) -> None:
        assert matrix.ndim == 2, "Matrix must be 2D"
        assert matrix.shape[0] == matrix.shape[1], "Matrix must be square"
        self._matrix = matrix

    @property
    def result(self) -> Any:
        return super().result

    def _compute(self) -> Any:
        return np.linalg.eig(self._matrix)
