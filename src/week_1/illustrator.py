from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from autocorrelation import Autocorrelation
from correlation import Correlation
from dataset import Dataset
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from mean import Mean
from numpy.typing import ArrayLike
from pca import PCA
from power_spectrum import PowerSpectrum

_PALETTE = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
    "#F0E442",
    "#000000",
]


class Illustrator:
    """Visualises a neural dataset of shape ``(n_trials, n_timesteps, n_neurons)``.

    :param dataset: The dataset to visualise.
    """

    def __init__(self, dataset: Dataset | ArrayLike) -> None:
        if isinstance(dataset, Dataset):
            self._dataset = dataset
        else:
            self._dataset = Dataset(np.asarray(dataset))

    @classmethod
    def from_array(cls, arr: ArrayLike) -> Illustrator:
        """Construct directly from a raw array of shape ``(n_trials, n_timesteps, n_neurons)``."""
        return cls(Dataset(np.asarray(arr)))

    # ------------------------------------------------------------------ text

    def summary(self) -> None:
        """Print per-neuron descriptive statistics to stdout."""
        ds = self._dataset
        mean_result = Mean(ds).result
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
        for i in range(ds.n_neurons):
            col = ds.observations[:, :, i]
            print(
                f"  {'Neuron ' + str(i):<14} "
                f"{mean_result.mean[:, i].mean():>8.3f} "
                f"{mean_result.sem[:, i].mean():>8.3f} "
                f"{col.min():>8.3f} "
                f"{col.max():>8.3f}"
            )
        print("=" * 60)

    # --------------------------------------------------------- raw traces

    def plot_neurons(
        self,
        trials: list[int] | None = None,
        neurons: list[int] | None = None,
        plot_mean: bool = True,
        plot_std: bool = False,
        population_centred: bool = False,
    ) -> Figure:
        """Plot raw trial traces for selected neurons.

        :param trials: Trial indices to plot. Defaults to all.
        :param neurons: Neuron indices to plot. Defaults to all.
        :param plot_mean: Overlay the population mean across neurons.
        :param plot_std: Overlay the std across neurons.
        :param population_centred: Subtract per-timestep neuron mean before plotting.
        :returns: The figure.
        """
        ds = self._dataset
        trials = list(range(ds.n_trials)) if trials is None else trials
        neurons = list(range(ds.n_neurons)) if neurons is None else neurons

        # population mean across neurons: (n_trials, n_timesteps)
        pop_mean = ds.observations.mean(axis=2)
        # population-centred data: subtract pop_mean from every neuron
        pop_centred = ds.observations - pop_mean[:, :, np.newaxis]

        fig, axes = plt.subplots(len(trials), 1, figsize=(10, 4 * len(trials)))
        if len(trials) == 1:
            axes = [axes]

        for ax, trial in zip(axes, trials):
            for n in neurons:
                y = (
                    pop_centred[trial, :, n]
                    if population_centred
                    else ds.observations[trial, :, n]
                )
                ax.plot(y, color=_PALETTE[n % len(_PALETTE)], label=f"Neuron {n}")

            if plot_mean:
                ax.plot(pop_mean[trial], color="black", lw=2, label="Pop. mean")

            if plot_std:
                std = ds.observations[trial, :, :][:, neurons].std(axis=1)
                ax.plot(std, color="red", lw=1.5, ls="--", label="Std")

            suffix = " (population-centred)" if population_centred else ""
            ax.set_title(f"Trial {trial + 1}{suffix}")
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Activity")
            ax.legend(fontsize=7, ncol=4, loc="best", frameon=False)

        fig.tight_layout()
        return fig

    # ------------------------------------------------- trial-averaged activity

    def plot_mean_and_std(
        self,
        neurons: list[int] | None = None,
        show_sem: bool = True,
        figsize: tuple[float, float] = (10, 4),
    ) -> Figure:
        """Plot trial-averaged activity ± SEM for selected neurons.

        :param neurons: Neuron indices to include. Defaults to all.
        :param show_sem: Whether to shade the SEM.
        :param figsize: Figure size as ``(width, height)``.
        :returns: The figure.
        """
        ds = self._dataset
        neurons = list(range(ds.n_neurons)) if neurons is None else neurons
        result = Mean(ds).result
        time = np.arange(ds.n_timesteps)

        fig, ax = plt.subplots(figsize=figsize)
        for k, n in enumerate(neurons):
            mean = result.mean[:, n]
            ax.plot(
                time, mean, color=_PALETTE[k % len(_PALETTE)], lw=2, label=f"Neuron {n}"
            )
            if show_sem:
                sem = result.sem[:, n]
                ax.fill_between(
                    time,
                    mean - sem,
                    mean + sem,
                    alpha=0.3,
                    color=_PALETTE[k % len(_PALETTE)],
                )

        ax.set_xlabel("Timestep")
        ax.set_ylabel("Activity")
        ax.set_title("Trial-averaged activity (± SEM)")
        ax.legend(fontsize=7, ncol=4, loc="best", frameon=False)
        fig.tight_layout()
        return fig

    # ------------------------------------------------- heatmaps

    def plot_heatmap(
        self,
        trial: int | None = None,
        figsize: tuple[float, float] = (8, 5),
    ) -> Figure:
        """Neuron × time heatmap.

        :param trial: Trial index. If ``None`` (default), uses the trial mean.
        :param figsize: Figure size as ``(width, height)``.
        :returns: The figure.
        """
        ds = self._dataset
        if trial is None:
            mat = Mean(ds).result.mean.T  # (n_neurons, n_timesteps)
            title = "Trial-averaged heatmap (neurons × time)"
        else:
            mat = ds.observations[trial].T  # (n_neurons, n_timesteps)
            title = f"Heatmap — Trial {trial + 1} (neurons × time)"

        vmax = np.abs(mat).max()
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", norm=norm)
        ax.set_yticks(range(ds.n_neurons))
        ax.set_yticklabels([f"Neuron {i}" for i in range(ds.n_neurons)], fontsize=7)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Neuron")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="Activity")
        fig.tight_layout()
        return fig

    def plot_neuron_heatmap(
        self,
        trials: list[int] | None = None,
        neurons: list[int] | None = None,
        population_centred: bool = False,
    ) -> Figure:
        """Per-trial heatmap of neuron activity (timesteps × neurons).

        :param trials: Trial indices to include. Defaults to all.
        :param neurons: Neuron indices to include. Defaults to all.
        :param population_centred: If ``True``, subtract per-timestep neuron mean.
        :returns: The figure.
        """
        ds = self._dataset
        trials = list(range(ds.n_trials)) if trials is None else trials
        neurons = list(range(ds.n_neurons)) if neurons is None else neurons

        pop_mean = ds.observations.mean(axis=2)
        pop_centred = ds.observations - pop_mean[:, :, np.newaxis]

        fig, axes = plt.subplots(len(trials), 1, figsize=(10, 4 * len(trials)))
        if len(trials) == 1:
            axes = [axes]

        for ax, trial in zip(axes, trials):
            mat = (
                pop_centred[trial, :, :][:, neurons]
                if population_centred
                else ds.observations[trial, :, :][:, neurons]
            )
            sns.heatmap(mat.T, cmap="viridis", cbar=True, ax=ax)
            suffix = " (population-centred)" if population_centred else ""
            ax.set_title(f"Trial {trial + 1}{suffix}")
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Neuron")

        fig.tight_layout()
        return fig

    # ----------------------------------------------- correlation

    def plot_correlation(
        self,
        trial: int | None = None,
        figsize: tuple[float, float] = (6, 5),
    ) -> Figure:
        """Plot the Pearson correlation matrix between neurons.

        :param trial: Trial index. If ``None`` (default), all trials are
            concatenated along the time axis.
        :param figsize: Figure size as ``(width, height)``.
        :returns: The figure.
        """
        ds = self._dataset
        corr = Correlation(ds, trial=trial).result
        n = ds.n_neurons
        labels = [f"Neuron {i}" for i in range(n)]

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

    # ------------------------------------------------------- PCA

    def plot_pca(
        self,
        n_components: int = 3,
        figsize: tuple[float, float] = (12, 4),
    ) -> Figure:
        """Plot top PC time courses and the PC1 vs PC2 state-space trajectory.

        :param n_components: Number of PCs to extract and display.
        :param figsize: Figure size as ``(width, height)``.
        :returns: The figure.
        """
        ds = self._dataset
        result = PCA(ds, n_components=n_components).result
        time = np.arange(ds.n_timesteps)

        fig = plt.figure(figsize=figsize)
        gs = GridSpec(1, 2, width_ratios=[2, 1], figure=fig)
        ax_l = fig.add_subplot(gs[0])
        ax_r = fig.add_subplot(gs[1])

        for k in range(result.scores.shape[1]):
            ax_l.plot(
                time,
                result.scores[:, k],
                color=_PALETTE[k % len(_PALETTE)],
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
        figsize: tuple[float, float] = (6, 4),
    ) -> Figure:
        """PCA scree plot showing per-component and cumulative variance explained.

        :param n_components: Number of components to show. Defaults to
            ``min(n_timesteps, n_neurons)``.
        :param figsize: Figure size as ``(width, height)``.
        :returns: The figure.
        """
        ds = self._dataset
        result = PCA(ds, n_components=n_components).result
        ev = result.explained_variance_ratio
        ks = np.arange(1, len(ev) + 1)

        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = ax1.twinx()

        ax1.bar(ks, ev * 100, color=_PALETTE[0], alpha=0.7, label="Individual")
        ax2.plot(
            ks, np.cumsum(ev) * 100, "o-", color=_PALETTE[1], lw=2, label="Cumulative"
        )
        ax2.axhline(90, color="grey", ls="--", lw=1, label="90% threshold")

        ax1.set_xlabel("Principal component")
        ax1.set_ylabel("Variance explained (%)", color=_PALETTE[0])
        ax2.set_ylabel("Cumulative variance (%)", color=_PALETTE[1])
        ax1.set_title("PCA — variance explained")

        lines1, labs1 = ax1.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="center right")

        fig.tight_layout()
        return fig

    # ------------------------------------------------------- autocorrelation

    def plot_autocorrelation(
        self,
        neurons: list[int] | None = None,
        max_lag: int | None = None,
        figsize: tuple[float, float] = (10, 4),
    ) -> Figure:
        """Plot the ACF for selected neurons, averaged across trials.

        :param neurons: Neuron indices. Defaults to all.
        :param max_lag: Maximum lag in timesteps. Defaults to ``n_timesteps // 2``.
        :param figsize: Figure size as ``(width, height)``.
        :returns: The figure.
        """
        ds = self._dataset
        neurons = list(range(ds.n_neurons)) if neurons is None else neurons
        result = Autocorrelation(ds, max_lag=max_lag).result

        fig, ax = plt.subplots(figsize=figsize)
        for k, n in enumerate(neurons):
            ax.plot(
                result.lags,
                result.acf[n],
                color=_PALETTE[k % len(_PALETTE)],
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

    # ------------------------------------------------------- power spectrum

    def plot_power_spectrum(
        self,
        neurons: list[int] | None = None,
        fs: float = 1.0,
        figsize: tuple[float, float] = (10, 4),
    ) -> Figure:
        """Plot Welch power spectral density for selected neurons, averaged across trials.

        :param neurons: Neuron indices. Defaults to all.
        :param fs: Sampling frequency. Defaults to ``1.0``.
        :param figsize: Figure size as ``(width, height)``.
        :returns: The figure.
        """
        ds = self._dataset
        neurons = list(range(ds.n_neurons)) if neurons is None else neurons
        result = PowerSpectrum(ds, fs=fs).result

        fig, ax = plt.subplots(figsize=figsize)
        for k, n in enumerate(neurons):
            ax.semilogy(
                result.frequencies,
                result.psd[n],
                color=_PALETTE[k % len(_PALETTE)],
                lw=2,
                label=f"Neuron {n}",
            )

        ax.set_xlabel("Frequency")
        ax.set_ylabel("Power spectral density (log scale)")
        ax.set_title("Power spectra (Welch, trial-averaged)")
        ax.legend(fontsize=7, ncol=4, frameon=False)
        fig.tight_layout()
        return fig

    # -------------------------------------------------- convenience

    def plot_all(self) -> None:
        """Run every visualisation method with default parameters."""
        self.summary()
        self.plot_neurons()
        self.plot_mean_and_std()
        self.plot_heatmap()
        self.plot_neuron_heatmap()
        self.plot_correlation()
        self.plot_pca()
        self.plot_variance_explained()
        self.plot_autocorrelation()
        self.plot_power_spectrum()
