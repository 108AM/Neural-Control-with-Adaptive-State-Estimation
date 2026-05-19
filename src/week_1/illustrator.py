from __future__ import annotations

import warnings
from functools import cached_property
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from autocorrelation import Autocorrelation
from coherence import Coherence
from correlation import Correlation
from cross_correlation import CrossCorrelation
from dataset import Dataset
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from mean import Mean, population_centre
from numpy.typing import ArrayLike
from pca import PCA
from power_spectrum import PowerSpectrum

IndexLike = Union[int, list[int], slice, range, "np.ndarray"]
"""Any value accepted as a dimension selector.

Supported forms:

* ``None`` — all indices in the dimension.
* ``int`` — single index (wrapped in a one-element list).
* ``list[int]`` — explicit list of indices.
* ``slice`` — e.g. ``slice(0, 8)`` or ``slice(None, None, 2)``.
* ``range`` — e.g. ``range(0, 8, 2)``.
* boolean ``np.ndarray`` of length *n* — selects indices where ``True``.
* integer ``np.ndarray`` — used as an index array.
"""


def _resolve_indices(sel: IndexLike | None, n: int) -> list[int]:
    """Normalise any index-like selector into a concrete list of ints.

    :param sel: Selector value. ``None`` expands to all ``n`` indices. See
        :data:`IndexLike` for the full set of accepted forms.
    :param n: Total number of elements in the dimension (used to expand
        ``None`` and resolve ``slice`` objects).
    :returns: List of non-negative integer indices. Order is preserved; no
        deduplication is performed.
    :raises IndexError: If any resolved index falls outside ``[0, n)``.
    :raises ValueError: If a boolean mask has a length other than *n*, or if
        *sel* is an unrecognised type.
    """
    if sel is None:
        return list(range(n))
    if isinstance(sel, (int, np.integer)):
        indices = [int(sel)]
    elif isinstance(sel, slice):
        indices = list(range(*sel.indices(n)))
    elif isinstance(sel, range):
        indices = list(sel)
    elif isinstance(sel, np.ndarray):
        if sel.dtype == bool:
            if len(sel) != n:
                raise ValueError(
                    f"Boolean mask length {len(sel)} does not match dimension size {n}."
                )
            indices = [int(i) for i in np.where(sel)[0]]
        else:
            indices = [int(i) for i in sel]
    elif isinstance(sel, list):
        indices = [int(i) for i in sel]
    else:
        raise ValueError(
            f"Unsupported index type {type(sel)!r}. Expected None, int, list[int], "
            "slice, range, or np.ndarray."
        )
    out_of_range = [i for i in indices if not (0 <= i < n)]
    if out_of_range:
        raise IndexError(
            f"Indices {out_of_range} are out of range for dimension of size {n}."
        )
    return indices


def _guard_subplots(
    indices: list[int], label: str, max_subplots: int, stacklevel: int = 3
) -> list[int]:
    """Truncate *indices* to *max_subplots* with a warning if needed.

    This prevents methods that create one subplot per element from producing
    figures with hundreds of panels when the dimension is large.

    :param indices: Already-resolved list of integer indices.
    :param label: Human-readable name of the dimension (e.g. ``"trials"``),
        used in the warning message.
    :param max_subplots: Maximum number of subplots allowed.
    :returns: *indices* unchanged, or its first *max_subplots* elements if
        the list was too long.
    """
    if len(indices) > max_subplots:
        warnings.warn(
            f"{len(indices)} {label} requested but max_subplots={max_subplots}. "
            f"Only the first {max_subplots} will be plotted. "
            "Increase max_subplots= to show more.",
            stacklevel=stacklevel,
        )
        return indices[:max_subplots]
    return indices


def _palette_color(k: int):
    """Return the k-th color from the current matplotlib prop cycle."""
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return colors[k % len(colors)]


class Illustrator:
    """Visualises a neural dataset of shape ``(n_trials, n_timesteps, n_neurons)``.

    :param dataset: The dataset to visualise. Any array-like is automatically
        wrapped in a :class:`~dataset.Dataset`.
    """

    def __init__(self, dataset: Dataset | ArrayLike) -> None:
        if isinstance(dataset, Dataset):
            self._dataset = dataset
        else:
            self._dataset = Dataset(np.asarray(dataset))

    @cached_property
    def _mean(self) -> Mean:
        return Mean(self._dataset)

    @classmethod
    def from_array(cls, arr: ArrayLike) -> Illustrator:
        """Construct directly from a raw array of shape ``(n_trials, n_timesteps, n_neurons)``."""
        return cls(Dataset(np.asarray(arr)))

    def summary(self) -> dict[str, dict]:
        """Print per-neuron descriptive statistics to stdout and return them.

        :returns: Dict keyed by ``"neuron_<i>"`` where each value is a dict
            with keys ``"mean"``, ``"sem"``, ``"min"``, and ``"max"``
            (all floats).
        """
        ds = self._dataset
        mean_result = self._mean.result
        print("=" * 60)
        print("Dataset summary")
        print("=" * 60)
        print(
            f"  Shape      : {ds.shape}  "
            f"(n_trials={ds.n_trials}, n_timesteps={ds.n_timesteps}, "
            f"n_neurons={ds.n_neurons})"
        )
        print()
        print(f"  {'Neuron':<14} {'Mean':>8} {'SEM':>8} {'Min':>8} {'Max':>8}")
        print("  " + "-" * 48)

        stats: dict[str, dict] = {}
        for i in range(ds.n_neurons):
            col = ds.observations[:, :, i]
            m = float(mean_result.mean[:, i].mean())
            s = float(mean_result.sem[:, i].mean())
            lo = float(col.min())
            hi = float(col.max())
            print(
                f"  {'Neuron ' + str(i):<14} {m:>8.3f} {s:>8.3f} {lo:>8.3f} {hi:>8.3f}"
            )
            stats[f"neuron_{i}"] = {"mean": m, "sem": s, "min": lo, "max": hi}

        print("=" * 60)
        return stats

    def plot_trials(
        self,
        trials: IndexLike | None = None,
        neurons: IndexLike | None = None,
        plot_mean: bool = True,
        plot_std: bool = False,
        population_centred: bool = False,
        figsize: tuple[float, float] | None = None,
        max_subplots: int = 20,
    ) -> Figure:
        """Plot raw trial traces for selected neurons.

        One subplot is drawn per selected trial.

        :param trials: Trials to plot. Accepts any :data:`IndexLike` form or
            ``None`` (all trials). Defaults to ``None``.
        :param neurons: Neurons to plot. Accepts any :data:`IndexLike` form
            or ``None`` (all neurons). Defaults to ``None``.
        :param plot_mean: Overlay the population mean across neurons.
        :param plot_std: Overlay the std across selected neurons.
        :param population_centred: Subtract the per-timestep neuron mean
            before plotting.
        :param figsize: Figure size as ``(width, height)``. Defaults to
            ``None``, which auto-scales to ``(10, 4 × n_selected_trials)``.
        :param max_subplots: Maximum number of trial subplots to draw. A
            :class:`UserWarning` is issued and the list is truncated when
            the selection exceeds this limit. Defaults to ``20``.
        :returns: The figure.
        """
        ds = self._dataset
        trial_indices = _guard_subplots(
            _resolve_indices(trials, ds.n_trials), "trials", max_subplots
        )
        neuron_indices = _resolve_indices(neurons, ds.n_neurons)
        _figsize = figsize if figsize is not None else (10, 4 * len(trial_indices))

        obs = ds.observations
        pop_mean = obs.mean(axis=2)  # (n_trials, n_timesteps)
        centred = population_centre(obs)  # (n_trials, n_timesteps, n_neurons)

        fig, axes = plt.subplots(len(trial_indices), 1, figsize=_figsize)
        if len(trial_indices) == 1:
            axes = [axes]

        for ax, trial in zip(axes, trial_indices):
            for k, n in enumerate(neuron_indices):
                y = centred[trial, :, n] if population_centred else obs[trial, :, n]
                ax.plot(y, color=_palette_color(k), label=f"Neuron {n}")

            if plot_mean:
                ax.plot(pop_mean[trial], color="black", lw=2, label="Pop. mean")

            if plot_std:
                _src = centred if population_centred else obs
                std = _src[trial, :, :][:, neuron_indices].std(axis=1)
                ax.plot(std, color="red", lw=1.5, ls="--", label="Std")

            suffix = " (population-centred)" if population_centred else ""
            ax.set_title(f"Trial {trial + 1}{suffix}")
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Activity")
            ax.legend(fontsize=7, ncol=4, loc="best", frameon=False)

        fig.tight_layout()
        return fig

    def plot_trial_average(
        self,
        neurons: IndexLike | None = None,
        show_sem: bool = True,
        population_centred: bool = False,
        figsize: tuple[float, float] = (10, 4),
    ) -> Figure:
        """Plot trial-averaged activity ± SEM for selected neurons.

        :param neurons: Neurons to include. Accepts any :data:`IndexLike`
            form or ``None`` (all neurons). Defaults to ``None``.
        :param show_sem: Whether to shade the SEM as a filled band.
        :param population_centred: Subtract the per-timestep cross-neuron
            mean from the trial-averaged traces before plotting. This shows
            each neuron's deviation from the population mean at each timestep.
        :param figsize: Figure size as ``(width, height)``.
        :returns: The figure.
        """
        ds = self._dataset
        neuron_indices = _resolve_indices(neurons, ds.n_neurons)
        result = self._mean.result
        time = np.arange(ds.n_timesteps)

        mean_mat = population_centre(result.mean) if population_centred else result.mean

        fig, ax = plt.subplots(figsize=figsize)
        for k, n in enumerate(neuron_indices):
            mean = mean_mat[:, n]
            ax.plot(time, mean, color=_palette_color(k), lw=2, label=f"Neuron {n}")
            if show_sem:
                sem = result.sem[:, n]
                ax.fill_between(
                    time,
                    mean - sem,
                    mean + sem,
                    alpha=0.3,
                    color=_palette_color(k),
                )

        suffix = " (population-centred)" if population_centred else ""
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Activity")
        ax.set_title(f"Trial-averaged activity (± SEM){suffix}")
        ax.legend(fontsize=7, ncol=4, loc="best", frameon=False)
        fig.tight_layout()
        return fig

    def plot_neuron_time_heatmap(
        self,
        trial: int | None = None,
        neurons: IndexLike | None = None,
        population_centred: bool = False,
        figsize: tuple[float, float] = (8, 5),
    ) -> Figure:
        """Neuron × time heatmap.

        :param trial: Trial index (0-based). If ``None`` (default), the
            trial-averaged data is used.
        :param neurons: Neurons to include. Accepts any :data:`IndexLike`
            form or ``None`` (all neurons). Defaults to ``None``.
        :param population_centred: Subtract the per-timestep cross-neuron
            mean from the matrix before rendering. This highlights relative
            deviations between neurons at each timestep.
        :param figsize: Figure size as ``(width, height)``.
        :returns: The figure.
        """
        ds = self._dataset
        neuron_indices = _resolve_indices(neurons, ds.n_neurons)

        if trial is None:
            mat = self._mean.result.mean.T  # (n_neurons, n_timesteps)
            title = "Trial-averaged heatmap (neurons × time)"
        else:
            mat = ds.observations[trial].T  # (n_neurons, n_timesteps)
            title = f"Heatmap — Trial {trial + 1} (neurons × time)"

        mat = mat[neuron_indices, :]  # (len(neuron_indices), n_timesteps)

        if population_centred:
            # centre across neurons at each timestep (axis=0 on transposed matrix)
            mat = mat - mat.mean(axis=0, keepdims=True)
            title += " (population-centred)"

        vmax = np.abs(mat).max()
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", norm=norm)
        ax.set_yticks(range(len(neuron_indices)))
        ax.set_yticklabels([f"Neuron {i}" for i in neuron_indices], fontsize=7)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Neuron")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="Activity")
        fig.tight_layout()
        return fig

    def plot_trial_heatmaps(
        self,
        trials: IndexLike | None = None,
        neurons: IndexLike | None = None,
        population_centred: bool = False,
        figsize: tuple[float, float] | None = None,
        max_subplots: int = 20,
    ) -> Figure:
        """Per-trial heatmap of neuron activity (timesteps × neurons).

        One subplot is drawn per selected trial.

        :param trials: Trials to include. Accepts any :data:`IndexLike` form
            or ``None`` (all trials). Defaults to ``None``.
        :param neurons: Neurons to include. Accepts any :data:`IndexLike`
            form or ``None`` (all neurons). Defaults to ``None``.
        :param population_centred: Subtract the per-timestep neuron mean
            before plotting.
        :param figsize: Figure size as ``(width, height)``. Defaults to
            ``None``, which auto-scales to ``(10, 4 × n_selected_trials)``.
        :param max_subplots: Maximum number of trial subplots to draw. A
            :class:`UserWarning` is issued and the list is truncated when
            the selection exceeds this limit. Defaults to ``20``.
        :returns: The figure.
        """
        ds = self._dataset
        trial_indices = _guard_subplots(
            _resolve_indices(trials, ds.n_trials), "trials", max_subplots
        )
        neuron_indices = _resolve_indices(neurons, ds.n_neurons)
        _figsize = figsize if figsize is not None else (10, 4 * len(trial_indices))

        obs = ds.observations
        centred = population_centre(obs)  # (n_trials, n_timesteps, n_neurons)

        fig, axes = plt.subplots(len(trial_indices), 1, figsize=_figsize)
        if len(trial_indices) == 1:
            axes = [axes]

        for ax, trial in zip(axes, trial_indices):
            mat = (
                centred[trial, :, :][:, neuron_indices]
                if population_centred
                else obs[trial, :, :][:, neuron_indices]
            )
            sns.heatmap(mat.T, cmap="viridis", cbar=True, ax=ax)
            suffix = " (population-centred)" if population_centred else ""
            ax.set_title(f"Trial {trial + 1}{suffix}")
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Neuron")

        fig.tight_layout()
        return fig

    def plot_correlation(
        self,
        trial: int | None = None,
        neurons: IndexLike | None = None,
        figsize: tuple[float, float] = (6, 5),
    ) -> Figure:
        """Plot the Pearson correlation matrix between neurons.

        :param trial: Trial index (0-based). If ``None`` (default), all
            trials are concatenated along the time axis before computing
            the correlation.
        :param neurons: Neurons to include. Accepts any :data:`IndexLike`
            form or ``None`` (all neurons). The full correlation matrix is
            computed first and then subsetted, so the result is always a
            valid symmetric correlation matrix. Defaults to ``None``.
        :param figsize: Figure size as ``(width, height)``.
        :returns: The figure.
        """
        ds = self._dataset
        neuron_indices = _resolve_indices(neurons, ds.n_neurons)
        corr = Correlation(ds, trial=trial).result  # (n_neurons, n_neurons)
        corr = corr[np.ix_(neuron_indices, neuron_indices)]
        n = len(neuron_indices)
        labels = [f"Neuron {i}" for i in neuron_indices]

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        title = "Inter-neuron correlation matrix"
        if trial is not None:
            title += f" (Trial {trial + 1})"
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="Pearson r")
        fig.tight_layout()
        return fig

    def plot_pca(
        self,
        n_components: int = 3,
        trials: IndexLike | None = None,
        figsize: tuple[float, float] = (12, 4),
    ) -> Figure:
        """Plot top PC time courses and the PC1 vs PC2 state-space trajectory.

        PCA is performed on the trial-averaged data of the selected trials.

        :param n_components: Number of PCs to extract and display.
        :param trials: Trials to average over. Accepts any :data:`IndexLike`
            form or ``None`` (all trials). Defaults to ``None``.
        :param figsize: Figure size as ``(width, height)``.
        :returns: The figure.
        """
        ds = self._dataset
        trial_indices = _resolve_indices(trials, ds.n_trials)
        subset = ds.select_trials(trial_indices)
        result = PCA(subset, n_components=n_components).result
        time = np.arange(ds.n_timesteps)

        fig = plt.figure(figsize=figsize)
        gs = GridSpec(1, 2, width_ratios=[2, 1], figure=fig)
        ax_l = fig.add_subplot(gs[0])
        ax_r = fig.add_subplot(gs[1])

        for k in range(result.scores.shape[1]):
            ax_l.plot(
                time,
                result.scores[:, k],
                color=_palette_color(k),
                lw=2,
                label=f"PC{k + 1} ({result.explained_variance_ratio[k] * 100:.1f}%)",
            )
        ax_l.set_xlabel("Timestep")
        ax_l.set_ylabel("PC score")
        ax_l.set_title("PCA — top principal components over time")
        ax_l.legend(fontsize=8, frameon=False)

        y2 = (
            result.scores[:, 1]
            if result.scores.shape[1] > 1
            else np.zeros(ds.n_timesteps)
        )
        sc = ax_r.scatter(
            result.scores[:, 0], y2, c=time, cmap="viridis", s=20, zorder=3
        )
        ax_r.plot(result.scores[:, 0], y2, color="grey", lw=0.8, alpha=0.5)
        ax_r.set_xlabel("PC1")
        ax_r.set_ylabel("PC2")
        ax_r.set_title("State-space trajectory (PC1 vs PC2)")
        fig.colorbar(sc, ax=ax_r, label="Timestep")

        fig.tight_layout()
        return fig

    def plot_variance_explained(
        self,
        n_components: int | None = None,
        trials: IndexLike | None = None,
        figsize: tuple[float, float] = (6, 4),
    ) -> Figure:
        """PCA scree plot showing per-component and cumulative variance explained.

        PCA is performed on the trial-averaged data of the selected trials.

        :param n_components: Number of components to show. Defaults to
            ``min(n_timesteps, n_neurons)``.
        :param trials: Trials to average over. Accepts any :data:`IndexLike`
            form or ``None`` (all trials). Defaults to ``None``.
        :param figsize: Figure size as ``(width, height)``.
        :returns: The figure.
        """
        ds = self._dataset
        trial_indices = _resolve_indices(trials, ds.n_trials)
        subset = ds.select_trials(trial_indices)
        result = PCA(subset, n_components=n_components).result
        ev = result.explained_variance_ratio
        ks = np.arange(1, len(ev) + 1)

        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = ax1.twinx()

        ax1.bar(ks, ev * 100, color=_palette_color(0), alpha=0.7, label="Individual")
        ax2.plot(
            ks,
            np.cumsum(ev) * 100,
            "o-",
            color=_palette_color(1),
            lw=2,
            label="Cumulative",
        )
        ax2.axhline(90, color="grey", ls="--", lw=1, label="90% threshold")

        ax1.set_xlabel("Principal component")
        ax1.set_ylabel("Variance explained (%)", color=_palette_color(0))
        ax2.set_ylabel("Cumulative variance (%)", color=_palette_color(1))
        ax1.set_title("PCA — variance explained")

        lines1, labs1 = ax1.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="center right")

        fig.tight_layout()
        return fig

    def plot_autocorrelation(
        self,
        neurons: IndexLike | None = None,
        trials: IndexLike | None = None,
        max_lag: int | None = None,
        figsize: tuple[float, float] = (10, 4),
    ) -> Figure:
        """Plot the ACF for selected neurons, averaged across selected trials.

        :param neurons: Neurons to plot. Accepts any :data:`IndexLike` form
            or ``None`` (all neurons). Defaults to ``None``.
        :param trials: Trials to average over. Accepts any :data:`IndexLike`
            form or ``None`` (all trials). Defaults to ``None``.
        :param max_lag: Maximum lag in timesteps. Defaults to
            ``n_timesteps // 2``.
        :param figsize: Figure size as ``(width, height)``.
        :returns: The figure.
        """
        ds = self._dataset
        trial_indices = _resolve_indices(trials, ds.n_trials)
        neuron_indices = _resolve_indices(neurons, ds.n_neurons)
        subset = ds.select_trials(trial_indices)
        result = Autocorrelation(subset, max_lag=max_lag).result

        fig, ax = plt.subplots(figsize=figsize)
        for k, n in enumerate(neuron_indices):
            ax.plot(
                result.lags,
                result.acf[n],
                color=_palette_color(k),
                lw=2,
                label=f"Neuron {n}",
            )

        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xlabel("Lag (timesteps)")
        ax.set_ylabel("Autocorrelation")
        ax.set_title("Autocorrelation function (trial-averaged)")
        ax.legend(fontsize=7, ncol=4, frameon=False)
        fig.tight_layout()
        return fig

    def plot_cross_correlation(
        self,
        neuron_a: int = 0,
        neuron_b: int = 1,
        trials: IndexLike | None = None,
        max_lag: int | None = None,
        figsize: tuple[float, float] = (10, 4),
    ) -> Figure:
        """Plot the cross-correlation function between two neurons, averaged across selected trials.

        :param neuron_a: Index of the reference neuron. Defaults to ``0``.
        :param neuron_b: Index of the target neuron. Defaults to ``1``.
        :param trials: Trials to average over. Accepts any :data:`IndexLike`
            form or ``None`` (all trials). Defaults to ``None``.
        :param max_lag: Maximum lag in timesteps (both positive and negative
            lags are shown). Defaults to ``n_timesteps // 2``.
        :param figsize: Figure size as ``(width, height)``.
        :returns: The figure.
        """
        ds = self._dataset
        trial_indices = _resolve_indices(trials, ds.n_trials)
        subset = ds.select_trials(trial_indices)
        result = CrossCorrelation(
            subset, neuron_a=neuron_a, neuron_b=neuron_b, max_lag=max_lag
        ).result

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(result.lags, result.ccf, color=_palette_color(0), lw=2)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.axvline(0, color="grey", lw=0.8, ls=":")
        ax.set_xlabel("Lag (timesteps)")
        ax.set_ylabel("Cross-correlation")
        ax.set_title(
            f"Cross-correlation: Neuron {neuron_a} → Neuron {neuron_b} (trial-averaged)"
        )
        fig.tight_layout()
        return fig

    def plot_power_spectrum(
        self,
        neurons: IndexLike | None = None,
        trials: IndexLike | None = None,
        fs: float = 1.0,
        figsize: tuple[float, float] = (10, 4),
    ) -> Figure:
        """Plot Welch power spectral density for selected neurons, averaged across selected trials.

        :param neurons: Neurons to plot. Accepts any :data:`IndexLike` form
            or ``None`` (all neurons). Defaults to ``None``.
        :param trials: Trials to average over. Accepts any :data:`IndexLike`
            form or ``None`` (all trials). Defaults to ``None``.
        :param fs: Sampling frequency. Defaults to ``1.0`` (normalised units).
        :param figsize: Figure size as ``(width, height)``.
        :returns: The figure.
        """
        ds = self._dataset
        trial_indices = _resolve_indices(trials, ds.n_trials)
        neuron_indices = _resolve_indices(neurons, ds.n_neurons)
        subset = ds.select_trials(trial_indices)
        result = PowerSpectrum(subset, fs=fs).result

        fig, ax = plt.subplots(figsize=figsize)
        for k, n in enumerate(neuron_indices):
            ax.semilogy(
                result.frequencies,
                result.psd[n],
                color=_palette_color(k),
                lw=2,
                label=f"Neuron {n}",
            )

        ax.set_xlabel("Frequency")
        ax.set_ylabel("Power spectral density (log scale)")
        ax.set_title("Power spectra (Welch, trial-averaged)")
        ax.legend(fontsize=7, ncol=4, frameon=False)
        fig.tight_layout()
        return fig

    def plot_coherence(
        self,
        neuron_pairs: list[tuple[int, int]] | None = None,
        trials: IndexLike | None = None,
        fs: float = 1.0,
        figsize: tuple[float, float] = (10, 4),
        max_pairs: int = 6,
    ) -> Figure:
        """Plot magnitude-squared coherence spectra for selected neuron pairs.

        Coherence ``Cxy(f) = |Sxy(f)|² / (Sxx(f) · Syy(f))`` measures the
        degree of linear coupling between two signals at each frequency. A
        value of 1 means the two neurons are perfectly linearly related at that
        frequency; 0 means they share no frequency content. This complements
        the univariate power spectrum: where :meth:`plot_power_spectrum` shows
        *which* frequencies each neuron oscillates at, this plot reveals
        *which* pairs of neurons share those oscillations.

        By default only the first ``max_pairs`` adjacent pairs (0→1, 1→2, …)
        are shown, keeping the plot legible. Pass *neuron_pairs* explicitly to
        focus on specific relationships — for example, pairs that showed high
        correlation in :meth:`plot_correlation`.

        :param neuron_pairs: Explicit list of ``(neuron_i, neuron_j)`` pairs to
            plot. Pass ``None`` (default) to auto-select the first *max_pairs*
            adjacent pairs.
        :param trials: Trials to average over. Accepts any :data:`IndexLike`
            form or ``None`` (all trials). Defaults to ``None``.
        :param fs: Sampling frequency in Hz. Defaults to ``1.0`` (normalised
            units). Set to the actual acquisition rate for a meaningful
            frequency axis.
        :param figsize: Figure size as ``(width, height)``.
        :param max_pairs: Maximum number of adjacent pairs shown when
            *neuron_pairs* is ``None``. Defaults to ``6``.
        :returns: The figure.
        """
        ds = self._dataset
        trial_indices = _resolve_indices(trials, ds.n_trials)
        subset = ds.select_trials(trial_indices)
        result = Coherence(subset, fs=fs).result

        if neuron_pairs is None:
            n = min(ds.n_neurons - 1, max_pairs)
            pairs = [(i, i + 1) for i in range(n)]
        else:
            pairs = neuron_pairs

        fig, ax = plt.subplots(figsize=figsize)
        for k, (i, j) in enumerate(pairs):
            ax.plot(
                result.frequencies,
                result.coherence[i, j],
                color=_palette_color(k),
                lw=2,
                label=f"Neuron {i} – {j}",
            )

        ax.set_xlabel("Frequency")
        ax.set_ylabel("Magnitude-squared coherence")
        ax.set_title("Coherence between neurons (Welch, trial-averaged)")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        return fig

    def plot_coherence_matrix(
        self,
        neurons: IndexLike | None = None,
        trials: IndexLike | None = None,
        fs: float = 1.0,
        f_low: float = 0.0,
        f_high: float | None = None,
        figsize: tuple[float, float] = (6, 5),
    ) -> Figure:
        """Heatmap of band-averaged magnitude-squared coherence between neurons.

        Collapses the full coherence spectrum to a single ``n_neurons ×
        n_neurons`` matrix by averaging over the frequency band
        ``[f_low, f_high]``. This gives an at-a-glance view of which neuron
        pairs share oscillatory activity — analogous to the Pearson correlation
        matrix from :meth:`plot_correlation`, but frequency-resolved: you can
        restrict the band to isolate, e.g., slow drift (low frequencies) or
        fast co-oscillations (high frequencies).

        The band-mean matrix is computed entirely inside :class:`~coherence.Coherence`
        and stored on :attr:`~coherence.CoherenceResult.band_matrix`, so this
        method contains no numerical logic.

        :param neurons: Neurons to include. Accepts any :data:`IndexLike` form
            or ``None`` (all neurons). Defaults to ``None``.
        :param trials: Trials to average over. Accepts any :data:`IndexLike`
            form or ``None`` (all trials). Defaults to ``None``.
        :param fs: Sampling frequency in Hz. Defaults to ``1.0`` (normalised
            units). Set to the actual acquisition rate for a meaningful band
            specification.
        :param f_low: Lower bound (inclusive) of the frequency band to average
            over. Defaults to ``0.0``.
        :param f_high: Upper bound (inclusive) of the frequency band to average
            over. Defaults to ``None`` (Nyquist — the full spectrum).
        :param figsize: Figure size as ``(width, height)``.
        :returns: The figure.
        """
        ds = self._dataset
        trial_indices = _resolve_indices(trials, ds.n_trials)
        neuron_indices = _resolve_indices(neurons, ds.n_neurons)
        subset = ds.select_trials(trial_indices)
        result = Coherence(subset, fs=fs, f_low=f_low, f_high=f_high).result

        mat = result.band_matrix[np.ix_(neuron_indices, neuron_indices)]
        n = len(neuron_indices)
        labels = [f"Neuron {i}" for i in neuron_indices]

        band_label = f"{f_low}–{f_high if f_high is not None else 'Nyquist'}"

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(f"Coherence matrix (band mean {band_label})")
        fig.colorbar(im, ax=ax, label="Magnitude-squared coherence")
        fig.tight_layout()
        return fig

    def plot_all(self, max_subplots: int = 20) -> dict[str, Figure]:
        """Run every visualisation method with default parameters.

        :param max_subplots: Maximum number of subplot panels for methods that
            produce one panel per trial (forwarded to :meth:`plot_trials` and
            :meth:`plot_trial_heatmaps`). Defaults to ``20``.
        :returns: Dict mapping method-name suffix to its :class:`Figure`. Keys
            are ``"trials"``, ``"trial_average"``, ``"neuron_time_heatmap"``,
            ``"trial_heatmaps"``, ``"correlation"``, ``"pca"``,
            ``"variance_explained"``, ``"autocorrelation"``,
            ``"cross_correlation"``, and ``"power_spectrum"``.
        """
        self.summary()
        return {
            "trials": self.plot_trials(max_subplots=max_subplots),
            "trial_average": self.plot_trial_average(),
            "neuron_time_heatmap": self.plot_neuron_time_heatmap(),
            "trial_heatmaps": self.plot_trial_heatmaps(max_subplots=max_subplots),
            "correlation": self.plot_correlation(),
            "pca": self.plot_pca(),
            "variance_explained": self.plot_variance_explained(),
            "autocorrelation": self.plot_autocorrelation(),
            "cross_correlation": self.plot_cross_correlation(),
            "power_spectrum": self.plot_power_spectrum(),
            "coherence": self.plot_coherence(),
            "coherence_matrix": self.plot_coherence_matrix(),
        }
