from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from dataset import Dataset
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from population_mean import PopulationMean


class Illustrator:
    """Visualises a neural dataset.

    :param dataset: The dataset to visualise.
    """

    def __init__(self, dataset: Dataset) -> None:
        self._dataset = dataset

    @classmethod
    def from_array(cls, arr: ArrayLike) -> Illustrator:
        """Construct directly from a raw array of shape ``(n_trials, n_timesteps, n_neurons)``."""
        return cls(Dataset(np.asarray(arr)))

    def plot_population_mean(self, ax: Axes | None = None) -> Figure:
        """Plot trial-averaged activity for each neuron over time.

        :param ax: Existing axes to draw into. Creates a new figure if ``None``.
        :returns: The figure containing the plot.
        """
        mean = PopulationMean(self._dataset).result  # (n_timesteps, n_neurons)
        time = np.arange(self._dataset.n_timesteps)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        else:
            raw = ax.get_figure()
            if not isinstance(raw, Figure):
                raise ValueError("ax has no associated Figure.")
            fig = raw

        for i in range(mean.shape[1]):
            ax.plot(time, mean[:, i], label=f"Neuron {i}")

        ax.set_xlabel("Timestep")
        ax.set_ylabel("Mean activity")
        ax.set_title("Population mean activity")
        ax.legend(fontsize=7, ncol=4, loc="best", frameon=False)

        return fig
