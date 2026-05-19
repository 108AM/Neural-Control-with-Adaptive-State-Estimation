from __future__ import annotations

from typing import NamedTuple

from analysis import Analysis
from dataset import Dataset
from numpy.typing import NDArray
from sklearn.decomposition import PCA as SklearnPCA


class PCAResult(NamedTuple):
    scores: NDArray  # (n_timesteps, n_components)
    explained_variance_ratio: NDArray  # (n_components,)
    components: NDArray  # (n_components, n_neurons)


class PCA(Analysis):
    """Principal component analysis on trial-averaged neural activity.

    The trial mean across repetitions is computed first, after which standard
    PCA is applied to the resulting population activity matrix.

    :param dataset: The dataset to analyse.
    :param n_components: Number of principal components to retain. Defaults to
        ``min(n_timesteps, n_neurons)``.

    :attr:`result` is a :class:`PCAResult`.
    """

    def __init__(self, dataset: Dataset, n_components: int | None = None) -> None:
        super().__init__(dataset)
        self._n_components = (
            n_components
            if n_components is not None
            else min(dataset.n_timesteps, dataset.n_neurons)
        )

    @property
    def result(self) -> PCAResult:
        return super().result  # type: ignore[return-value]

    def _compute(self) -> PCAResult:
        trial_mean = self._dataset.observations.mean(axis=0)

        pca = SklearnPCA(n_components=self._n_components)
        scores = pca.fit_transform(trial_mean)

        return PCAResult(
            scores=scores,
            explained_variance_ratio=pca.explained_variance_ratio_,
            components=pca.components_,
        )
