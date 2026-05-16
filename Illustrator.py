import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

class Illustrator:
    """
    A class to visualize neural activity data.
    Please finish the implementation and annotations of this class.
    """
    # __slots__ = ['observation', 'trial_cnt', 'timestep_cnt', 'neuron_cnt', 'mean']

    def __init__(self, observation: np.ndarray):
        """
        Initialize the Illustrator with the given observation data.
        Accepts a 3D numpy array of shape (Trials, Timepoints, Neurons) and stores it for visualization.
        Parameters:
            observation (np.ndarray): A 3D array containing neural activity data with dimensions (Trials, Timepoints, Neurons).
        """ 
        self.observation = observation
        self.trial_cnt, self.timestep_cnt, self.neuron_cnt = observation.shape
        self.mean = np.mean(observation, axis=2)  # Mean activity across neurons for each trial and timepoint
        self.observations_centered = self.observation - self.mean[:, :, np.newaxis]  # Center the data by subtracting the mean activity

    def plot_neuron_activity(self, trials = None, neurons = None, plot_mean = False, plot_std = False, centered = False):
        """Plot the activity of all neurons for each trial.
        Parameters:
            trials: list of trial indices to plot (default: all trials)
            neurons: list of neuron indices to plot (default: all neurons)
            plot_mean: boolean indicating whether to plot the mean activity (default: False)
            plot_std: boolean indicating whether to plot the standard deviation (default: False)
            centered: boolean indicating whether to plot centered data (remove mean) (default: False)
        """
        if trials is None:
            trials = list(range(self.trial_cnt))
        if neurons is None:
            neurons = list(range(self.neuron_cnt))

        fig, axes = plt.subplots(len(trials), 1, figsize=(10, 6 * len(trials)))

        for trial_index, trial in enumerate(trials):
            subplot_title = f'Trial {trial + 1}'
            for neuron in neurons:
                if centered:
                    axes[trial_index].plot(self.observations_centered[trial, :, neuron], label=f'Neuron {neuron}')
                else:
                    axes[trial_index].plot(self.observation[trial, :, neuron], label=f'Neuron {neuron}')
            if plot_mean:
                axes[trial_index].plot(self.mean[trial, :], label='Mean Activity', color='black', linewidth=3)
            if plot_std:
                std_activity = np.std(self.observation[trial, :, neurons], axis=1)
                axes[trial_index].plot(std_activity, label='Standard Deviation', color='red', linewidth=2)
            axes[trial_index].set_title(subplot_title)
            axes[trial_index].set_xlabel('Timepoints')
            axes[trial_index].set_ylabel('Activity')
            axes[trial_index].legend()
        plt.show()

    def plot_neuron_heatmap(self, trials=None, neurons=None):
        """Plot a heatmap of neuron activity across all trials and timepoints after centering.
        
        Parameters:
            trials: list of trial indices to include in the heatmap (default: all trials)
            neurons: list of neuron indices to include in the heatmap (default: all neurons)
        """
        if trials is None:
            trials = list(range(self.trial_cnt))
        if neurons is None:
            neurons = list(range(self.neuron_cnt))

        fig, axes = plt.subplots(len(trials), 1, figsize=(10, 6 * len(trials)))

        for trial_index, trial in enumerate(trials):
            subplot_title = f'Trial {trial + 1} (Centered)'
            centered_activity = self.observations_centered[trial, :, neurons]
            sns.heatmap(centered_activity, cmap='viridis', cbar=True, ax=axes[trial_index]) 
            axes[trial_index].set_title(subplot_title)
            axes[trial_index].set_xlabel('Timepoints')
            axes[trial_index].set_ylabel('Centered Activity (Neurons)')
        plt.show()
        # data_reshaped = (self.observation - self.mean[:, :, np.newaxis]).reshape(-1, self.neuron_cnt)
        # plt.figure(figsize=(12, 8))
        # sns.heatmap(data_reshaped.T, cmap='viridis', cbar=True)
        # plt.title('Neuron Activity Heatmap (Neurons x Timepoints)')
        # plt.xlabel('Timepoints (Trials concatenated)')
        # plt.ylabel('Neurons')
        # plt.show()

    def pca(self, n_components: int = None, plot: bool = True):
        """Reduce data to latent space using PCA and plot the dynamics.
        returns the latent representation of the data in the reduced PCA space."""
        data_centered = self.observation - self.mean[:, :, np.newaxis]  # Center the data by subtracting the mean activity
        data_reshaped = data_centered.reshape(-1, self.neuron_cnt)
        if n_components is None:
            n_components = self.neuron_cnt  # Keep all components if n_components is not specified
        pca = PCA(n_components=n_components)
        latent = pca.fit_transform(data_reshaped)
        latent = latent.reshape(self.trial_cnt, self.timestep_cnt, -1)
        
        if plot:
            plt.figure(figsize=(8, 5))
            plt.plot(range(1, n_components + 1), np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
            plt.axhline(y=0.9, color='r', linestyle='-', label='90% Variance threshold')
            plt.xlabel('Number of Components')
            plt.ylabel('Cumulative Explained Variance')
            plt.title('PCA Explained Variance (Scree Plot)')
            plt.tight_layout()

            fig, axes = plt.subplots(n_components, 1, figsize=(10, 3 * n_components), sharex=True)
            if n_components == 1:
                axes = [axes]
                
            for i, ax in enumerate(axes):
                for trial_index in range(self.trial_cnt):
                    ax.plot(latent[trial_index, :, i], label=f'Trial {trial_index + 1}')
                ax.set_ylabel(f'PC {i + 1}')
                if i == 0:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    
            axes[-1].set_xlabel('Timepoints')
            plt.suptitle(f'Latent Dynamics ({n_components} PCA Components)')
            plt.grid(True)
            plt.show()
        
        return latent
        