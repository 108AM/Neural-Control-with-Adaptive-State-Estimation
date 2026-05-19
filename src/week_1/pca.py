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

    The trial mean is computed first. The per-timestep neuron mean is then
    subtracted (demeaning across neurons at each timestep) before fitting PCA,
    so that components capture differential structure rather than the shared
    mean response.

    :param dataset: The dataset to analyse.
    :param n_components: Number of principal components to retain. Defaults to
        ``min(n_timesteps, n_neurons)``.

    :attr:`result` is a :class:`PCAResult`.
    """

    def __init__(self, dataset: Dataset, n_components: int | None = None) -> None:
        super().__init__(dataset)
        self._n_components = n_components or min(dataset.n_timesteps, dataset.n_neurons)

    @property
    def result(self) -> PCAResult:
        return super().result  # type: ignore[return-value]

    def _compute(self) -> PCAResult:
        trial_mean = self._dataset.observations.mean(axis=0)  # (n_timesteps, n_neurons)
        demeaned = trial_mean - trial_mean.mean(axis=1, keepdims=True)
        pca = SklearnPCA(n_components=self._n_components)
        scores = pca.fit_transform(demeaned)  # (n_timesteps, n_components)
        return PCAResult(
            scores=scores,
            explained_variance_ratio=pca.explained_variance_ratio_,
            components=pca.components_,
        )
