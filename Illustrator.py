"""
Illustrator.py
==============
A class for exploring, summarising, and visualising neural population datasets.

Expected data format
--------------------
A 3-D NumPy array with shape (Trials, Timepoints, Neurons).

Quick-start
-----------
    import numpy as np
    from Illustrator import Illustrator

    data = np.load("ExampleDataset.npy")   # shape (5, 60, 16)
    ill  = Illustrator(data)

    ill.summary()                 # print dataset statistics
    ill.plot_neurons()            # raw traces per trial, all neurons
    ill.plot_mean_and_std()       # trial-averaged activity ± SEM
    ill.plot_heatmap()            # neuron × time heatmap (trial-averaged)
    ill.plot_neuron_heatmap()     # per-trial heatmap, optionally centred
    ill.plot_correlation()        # inter-neuron correlation matrix
    ill.plot_pca()                # top PCs of population activity
    ill.plot_variance_explained() # PCA scree plot
    ill.plot_autocorrelation()    # per-neuron autocorrelation
    ill.plot_power_spectrum()     # per-neuron power spectra
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from sklearn.decomposition import PCA
import seaborn as sns
from scipy.signal import welch
import warnings

# ── colour palette (colour-blind friendly) ──────────────────────────────────
_PALETTE = [
    "#0072B2", "#E69F00", "#009E73", "#CC79A7",
    "#56B4E9", "#D55E00", "#F0E442", "#000000",
]


class Illustrator:
    """
    Explores and visualises a neural dataset of shape (Trials, Timepoints, Neurons).

    Parameters
    ----------
    data : np.ndarray, shape (T, N_t, N_n)
        T   – number of trials
        N_t – number of time-points per trial
        N_n – number of neurons (observed dimensions)
    dt : float, optional
        Sampling interval (default 1.0 — arbitrary time units).
    neuron_labels : list of str, optional
        Names for each neuron channel.  Defaults to ["Neuron 0", "Neuron 1", …].
    """

    # ------------------------------------------------------------------ init
    def __init__(self, data: np.ndarray, dt: float = 1.0,
                 neuron_labels: list = None):
        if data.ndim != 3:
            raise ValueError(
                f"data must be 3-D (Trials, Timepoints, Neurons); got {data.ndim}-D.")

        self.data = data                                    # (T, N_t, N_n)
        self.n_trials, self.n_time, self.n_neurons = data.shape
        self.dt   = dt
        self.time = np.arange(self.n_time) * dt

        if neuron_labels is None:
            self.neuron_labels = [f"Neuron {i}" for i in range(self.n_neurons)]
        else:
            if len(neuron_labels) != self.n_neurons:
                raise ValueError(
                    f"len(neuron_labels) = {len(neuron_labels)} does not match "
                    f"the number of neurons ({self.n_neurons}).")
            self.neuron_labels = list(neuron_labels)

        # ── derived representations ──────────────────────────────────────────
        # Average across trials → (N_t, N_n)
        self.trial_averaged_data = data.mean(axis=0)

        # Alias used by several methods
        self.mean_activity = self.trial_averaged_data

        # Mean across neurons at each (trial, timepoint) → (T, N_t)
        # NOTE: axis=2 averages over neurons; axis=1 would average over timepoints
        self.population_averaged_data = data.mean(axis=2)

        # Subtract the per-timepoint population mean from every neuron → (T, N_t, N_n)
        # [:, :, np.newaxis] broadcasts (T, N_t) → (T, N_t, 1) against (T, N_t, N_n)
        self.population_centred_data = data - self.population_averaged_data[:, :, np.newaxis]

    # ----------------------------------------------------------- text summary
    def summary(self) -> None:
        """
        Print basic statistics of the dataset:
        shape, per-neuron mean / standard deviation / min / max,
        and global signal-to-noise ratio (mean / std across time).
        """
        print("=" * 60)
        print("Dataset summary")
        print("=" * 60)
        print(f"  Shape            : {self.data.shape}  "
              f"(Trials={self.n_trials}, Timepoints={self.n_time}, "
              f"Neurons={self.n_neurons})")
        print(f"  Sampling interval: dt = {self.dt}")
        print(f"  Total duration   : {self.n_time * self.dt:.3g} time-units")
        print()
        print(f"  {'Neuron':<14} {'Mean':>8} {'Std':>8} {'Min':>8} "
              f"{'Max':>8} {'SNR':>8}")
        print("  " + "-" * 56)
        for i, label in enumerate(self.neuron_labels):
            col  = self.data[:, :, i]
            mean = col.mean()
            std  = col.std()
            snr  = abs(mean) / std if std > 0 else np.nan
            print(f"  {label:<14} {mean:>8.3f} {std:>8.3f} "
                  f"{col.min():>8.3f} {col.max():>8.3f} {snr:>8.3f}")
        print("=" * 60)

    # ------------------------------------------------------- raw trial traces
    def plot_neurons(self, trials=None, neurons=None,
                     plot_mean=False, plot_std=False,
                     population_centred=False):
        """
        Plot individual trial traces for a chosen set of neurons.

        Parameters
        ----------
        trials : list of int, optional
            Indices of trials to plot.  Defaults to all trials.
        neurons : list of int, optional
            Indices of neurons to plot.  Defaults to all neurons.
        plot_mean : bool
            Overlay the population mean (mean across neurons) at each timepoint.
        plot_std : bool
            Overlay the standard deviation across neurons at each timepoint.
        population_centred : bool
            If True, plot population-centred data (mean across neurons removed
            at every timepoint) rather than raw activity.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if trials is None:
            trials = list(range(self.n_trials))
        if neurons is None:
            neurons = list(range(self.n_neurons))

        fig, axes = plt.subplots(len(trials), 1, figsize=(10, 6 * len(trials)))
        if len(trials) == 1:
            axes = [axes]

        for trial_index, trial in enumerate(trials):
            subplot_title = (f"Trial {trial + 1} (population-centred)"
                             if population_centred else f"Trial {trial + 1}")

            for neuron in neurons:
                # population_averaged_data is (T, N_t) — 2-D, cannot be indexed
                # by neuron.  Use population_centred_data (T, N_t, N_n) instead.
                y = (self.population_centred_data[trial, :, neuron]
                     if population_centred
                     else self.data[trial, :, neuron])
                axes[trial_index].plot(y, label=f"Neuron {neuron}")

            if plot_mean:
                # population_averaged_data[trial] → (N_t,): mean across neurons
                axes[trial_index].plot(
                    self.population_averaged_data[trial],
                    label="Mean activity", color="black", linewidth=3)

            if plot_std:
                std_activity = np.std(self.data[trial, :, :][:, neurons], axis=1)
                axes[trial_index].plot(
                    std_activity,
                    label="Std dev", color="red", linewidth=2)

            axes[trial_index].set_title(subplot_title)
            axes[trial_index].set_xlabel("Timepoints")
            axes[trial_index].set_ylabel("Activity")
            axes[trial_index].legend()

        plt.tight_layout()
        plt.show()
        return fig

    # ------------------------------------------------- trial-averaged activity
    def plot_mean_and_std(self, neurons: list = None,
                          show_sem: bool = True,
                          figsize: tuple = (10, 4)) -> plt.Figure:
        """
        Plot trial-averaged activity (± SEM) for selected neurons.

        Parameters
        ----------
        neurons : list of int, optional
            Neuron indices to include.  Defaults to all.
        show_sem : bool
            Whether to shade the standard error of the mean.
        figsize : (width, height)

        Returns
        -------
        matplotlib.figure.Figure
        """
        neurons = list(range(self.n_neurons)) if neurons is None else neurons

        fig, ax = plt.subplots(figsize=figsize)
        sem = self.data[:, :, neurons].std(axis=0) / np.sqrt(self.n_trials)  # (N_t, len(neurons))

        for k, ni in enumerate(neurons):
            mean = self.mean_activity[:, ni]
            ax.plot(self.time, mean, lw=2, label=self.neuron_labels[ni])
            if show_sem:
                ax.fill_between(self.time,
                                mean - sem[:, k],
                                mean + sem[:, k], alpha=0.5)

        ax.set_xlabel("Time")
        ax.set_ylabel("Activity")
        ax.set_title("Trial-averaged activity (± SEM)")
        ax.legend(fontsize=7, ncol=4, loc="upper right")
        fig.tight_layout()
        plt.show()
        return fig

    # ------------------------------------------------------- heatmap (neuron × time)
    def plot_heatmap(self, trial: int = None,
                     figsize: tuple = (10, 5)) -> plt.Figure:
        """
        Neuron × Time heatmap of activity.

        Parameters
        ----------
        trial : int or None
            If given, plot data from that single trial.
            If None (default), plot the trial-averaged data.
        figsize : (width, height)

        Returns
        -------
        matplotlib.figure.Figure
        """
        if trial is None:
            mat   = self.mean_activity.T
            title = "Trial-averaged heatmap (neurons × time)"
        else:
            if trial < 0 or trial >= self.n_trials:
                raise ValueError(f"trial must be in [0, {self.n_trials - 1}].")
            mat   = self.data[trial, :, :].T
            title = f"Heatmap — Trial {trial} (neurons × time)"

        vmax = np.abs(mat).max()
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", norm=norm,
                       extent=[self.time[0], self.time[-1],
                               self.n_neurons - 0.5, -0.5])
        ax.set_yticks(range(self.n_neurons))
        ax.set_yticklabels(self.neuron_labels, fontsize=7)
        ax.set_xlabel("Time")
        ax.set_ylabel("Neuron")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="Activity")
        fig.tight_layout()
        plt.show()
        return fig

    def plot_neuron_heatmap(self, trials=None, neurons=None,
                            population_centred=True):
        """
        Per-trial heatmap of neuron activity (Timepoints × Neurons).

        Parameters
        ----------
        trials : list of int, optional
            Trial indices to include.  Defaults to all.
        neurons : list of int, optional
            Neuron indices to include.  Defaults to all.
        population_centred : bool
            If True (default), plot population-centred data.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if trials is None:
            trials = list(range(self.n_trials))
        if neurons is None:
            neurons = list(range(self.n_neurons))

        fig, axes = plt.subplots(len(trials), 1, figsize=(10, 6 * len(trials)))
        if len(trials) == 1:
            axes = [axes]

        for trial_index, trial in enumerate(trials):
            subplot_title = (f"Trial {trial + 1} (population-centred)"
                             if population_centred else f"Trial {trial + 1}")
            mat = (self.population_centred_data[trial, :, :][:, neurons]
                   if population_centred
                   else self.data[trial, :, :][:, neurons])
            sns.heatmap(mat.T, cmap="viridis", cbar=True, ax=axes[trial_index])
            axes[trial_index].set_title(subplot_title)
            axes[trial_index].set_xlabel("Timepoints")
            axes[trial_index].set_ylabel("Neurons")

        plt.tight_layout()
        plt.show()
        return fig

    # ----------------------------------------------- inter-neuron correlation
    def plot_correlation(self, trial: int = None,
                         figsize: tuple = (6, 5)) -> plt.Figure:
        """
        Plot the Pearson correlation matrix between neurons.

        Parameters
        ----------
        trial : int or None
            If given, compute correlation from that trial only.
            Otherwise uses all trials concatenated along the time axis.
        figsize : (width, height)

        Returns
        -------
        matplotlib.figure.Figure
        """
        if trial is None:
            mat = self.data.reshape(-1, self.n_neurons)
        else:
            mat = self.data[trial]

        corr = np.corrcoef(mat.T)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(self.n_neurons))
        ax.set_yticks(range(self.n_neurons))
        ax.set_xticklabels(self.neuron_labels, rotation=90, fontsize=7)
        ax.set_yticklabels(self.neuron_labels, fontsize=7)
        ax.set_title("Inter-neuron correlation matrix"
                     + ("" if trial is None else f" (Trial {trial})"))
        fig.colorbar(im, ax=ax, label="Pearson r")
        fig.tight_layout()
        plt.show()
        return fig

    # ------------------------------------------------------- internal helpers
    def _demean(self, mat: np.ndarray) -> np.ndarray:
        """
        Remove the population mean at each timestep.

        At every time-point t the mean activity across all neurons is subtracted:

            mat_centred[t, :] = mat[t, :] - mean(mat[t, :])

        This must be done before PCA so that the first principal component
        reflects the dominant *pattern of differential activation* across
        neurons, rather than simply the direction of the shared mean response
        (which would otherwise capture the most variance by construction).

        Parameters
        ----------
        mat : np.ndarray, shape (N_t, N_n)

        Returns
        -------
        np.ndarray, shape (N_t, N_n)  — mean-subtracted along the neuron axis
        """
        return mat - mat.mean(axis=1, keepdims=True)

    # ------------------------------------------------------- PCA trajectories
    def plot_pca(self, n_components: int = 3,
                 figsize: tuple = (12, 4)) -> plt.Figure:
        """
        Reduce the trial-averaged, population-centred activity to its top
        principal components and plot each PC over time.

        The population mean across neurons is removed at each timestep before
        fitting (see ``_demean``), so that PCA captures differential structure
        rather than the shared mean response.

        Parameters
        ----------
        n_components : int
            Number of PCs to extract and display.
        figsize : (width, height)

        Returns
        -------
        matplotlib.figure.Figure
        """
        n_components = min(n_components, self.n_neurons, self.n_time)
        centred  = self._demean(self.trial_averaged_data)   # (N_t, N_n)
        pca      = PCA(n_components=n_components)
        scores   = pca.fit_transform(centred)               # (N_t, n_components)

        fig = plt.figure(figsize=figsize)
        gs  = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

        # ── left: PC time courses ────────────────────────────────────────
        ax_l = fig.add_subplot(gs[0])
        for k in range(n_components):
            ax_l.plot(self.time, scores[:, k],
                      color=_PALETTE[k % len(_PALETTE)], lw=2,
                      label=f"PC{k+1} ({pca.explained_variance_ratio_[k]*100:.1f}%)")
        ax_l.set_xlabel("Time")
        ax_l.set_ylabel("PC score")
        ax_l.set_title("PCA — top principal components over time")
        ax_l.legend(fontsize=8)

        # ── right: PC1 vs PC2 trajectory (coloured by time) ─────────────
        ax_r = fig.add_subplot(gs[1])
        y2 = scores[:, 1] if n_components > 1 else np.zeros(self.n_time)
        sc = ax_r.scatter(scores[:, 0], y2, c=self.time, cmap="viridis", s=20, zorder=3)
        ax_r.plot(scores[:, 0], y2, color="grey", lw=0.8, alpha=0.5)
        ax_r.set_xlabel("PC1")
        ax_r.set_ylabel("PC2")
        ax_r.set_title("State-space trajectory (PC1 vs PC2)")
        fig.colorbar(sc, ax=ax_r, label="Time")

        fig.tight_layout()
        plt.show()
        return fig

    # ------------------------------------------------------- variance explained
    def plot_variance_explained(self, n_components: int = None,
                                figsize: tuple = (6, 4)) -> plt.Figure:
        """
        Plot the fraction and cumulative variance explained by PCA using the
        trial-averaged data centred by removing the population mean at each
        time-point (see ``_demean``).

        Parameters
        ----------
        n_components : int or None
            Number of components to show.  Defaults to min(neurons, time).
        figsize : (width, height)

        Returns
        -------
        matplotlib.figure.Figure
        """
        n_max = min(self.n_neurons, self.n_time)
        if n_components is not None:
            n_max = min(n_components, n_max)

        centred = self._demean(self.trial_averaged_data)
        pca     = PCA(n_components=n_max)
        pca.fit(centred)
        ev = pca.explained_variance_ratio_
        ks = np.arange(1, n_max + 1)

        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = ax1.twinx()

        ax1.bar(ks, ev * 100, color=_PALETTE[0], alpha=0.7, label="Individual")
        ax2.plot(ks, np.cumsum(ev) * 100, "o-",
                 color=_PALETTE[1], lw=2, label="Cumulative")
        ax2.axhline(90, color="grey", ls="--", lw=1, label="90 % threshold")

        ax1.set_xlabel("Principal component")
        ax1.set_ylabel("Variance explained (%)", color=_PALETTE[0])
        ax2.set_ylabel("Cumulative variance (%)", color=_PALETTE[1])
        ax1.set_title("PCA — variance explained")

        lines1, labs1 = ax1.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="center right")

        fig.tight_layout()
        plt.show()
        return fig

    # ------------------------------------------------------- autocorrelation
    def plot_autocorrelation(self, neurons: list = None,
                             max_lag: int = None,
                             figsize: tuple = (10, 4),
                             return_max_lags: bool = True):
        """
        Plot the autocorrelation function (ACF) for selected neurons,
        averaged across trials.

        Parameters
        ----------
        neurons : list of int, optional
            Neuron indices.  Defaults to all.
        max_lag : int or None
            Maximum lag to display (in time-steps).  Defaults to N_t // 2.
        figsize : (width, height)
        return_max_lags : bool
            If True, returns a tuple of (Figure, dict mapping neuron to lag of max correlation).
            Excludes lag 0, which is always 1.0 by definition.

        Returns
        -------
        matplotlib.figure.Figure (or tuple of Figure, dict if return_max_lags is True)
        """
        neurons = list(range(self.n_neurons)) if neurons is None else neurons
        max_lag = self.n_time // 2 if max_lag is None else max_lag
        lags    = np.arange(0, max_lag + 1) * self.dt

        max_correlation_lags = {}

        fig, ax = plt.subplots(figsize=figsize)
        for k, ni in enumerate(neurons):
            acfs = []
            for t in range(self.n_trials):
                x  = self.data[t, :, ni] - self.data[t, :, ni].mean()
                ac = np.correlate(x, x, mode="full")
                ac = ac[len(ac) // 2:]
                ac = ac[:max_lag + 1] / ac[0]
                acfs.append(ac)
            
            mean_acf = np.mean(acfs, axis=0)
            
            # Find the lag with maximum correlation (ignoring lag 0, since that is measuring correlation with itself at the same time, which mathematically is always 1.0)
            if len(mean_acf) > 1:
                best_lag_idx = np.argmax(mean_acf[1:]) + 1
                max_correlation_lags[self.neuron_labels[ni]] = lags[best_lag_idx]
            else:
                 max_correlation_lags[self.neuron_labels[ni]] = None

            ax.plot(lags, mean_acf,
                    color=_PALETTE[k % len(_PALETTE)],
                    lw=2, label=self.neuron_labels[ni])

        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xticks(lags)
        ax.grid(True, axis='x', linestyle='--', alpha=0.5)
        ax.set_xlabel("Lag (time units)")
        ax.set_ylabel("Autocorrelation")
        ax.set_title("Autocorrelation function (trial-averaged)")
        ax.legend(fontsize=7, ncol=4)
        fig.tight_layout()
        plt.show()
        
        if return_max_lags:
            return fig, max_correlation_lags
        return fig

    # ------------------------------------------------------- power spectrum
    def plot_power_spectrum(self, neurons: list = None,
                            fs: float = None,
                            figsize: tuple = (10, 4)) -> plt.Figure:
        """
        Plot the power spectral density (Welch's method) for selected neurons,
        averaged across trials.

        Parameters
        ----------
        neurons : list of int, optional
            Neuron indices.  Defaults to all.
        fs : float or None
            Sampling frequency in Hz.  Defaults to 1/dt.
        figsize : (width, height)

        Returns
        -------
        matplotlib.figure.Figure
        """
        neurons = list(range(self.n_neurons)) if neurons is None else neurons
        fs      = 1.0 / self.dt if fs is None else fs

        fig, ax = plt.subplots(figsize=figsize)
        for k, ni in enumerate(neurons):
            psds = []
            for t in range(self.n_trials):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    f, psd = welch(self.data[t, :, ni], fs=fs,
                                   nperseg=min(32, self.n_time))
                psds.append(psd)
            ax.semilogy(f, np.mean(psds, axis=0),
                        color=_PALETTE[k % len(_PALETTE)],
                        lw=2, label=self.neuron_labels[ni])

        ax.set_xlabel("Frequency")
        ax.set_ylabel("Power spectral density (log scale)")
        ax.set_title("Power spectra (Welch, trial-averaged)")
        ax.legend(fontsize=7, ncol=4)
        fig.tight_layout()
        plt.show()
        return fig

    # -------------------------------------------------- convenience: all plots
    def plot_all(self) -> None:
        """
        Run every visualisation method in sequence — a one-stop overview.
        """
        self.summary()
        self.plot_neurons()
        self.plot_mean_and_std()
        self.plot_heatmap()
        self.plot_neuron_heatmap()
        self.plot_correlation()
        self.plot_pca()
        self.plot_variance_explained()
        fig, max_lags = self.plot_autocorrelation(return_max_lags=True)
        for neuron, best_lag in max_lags.items():
            print(f"{neuron} peaks at a lag of {best_lag} time units.")
        self.plot_power_spectrum()