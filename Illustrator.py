import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

class Illustrator:
    """
    A class to visualize neural activity data.
    Please finish the implementation and annotations of this class.
    """
    __slots__ = ['observation', 'trial_cnt', 'timestep_cnt', 'neuron_cnt']

    def __init__(self, observation: np.ndarray):
        """
        Initialize the Illustrator with the given observation data.
        Accepts a 3D numpy array of shape (Trials, Timepoints, Neurons) and stores it for visualization.
        Parameters:
            observation (np.ndarray): A 3D array containing neural activity data with dimensions (Trials, Timepoints, Neurons).
        """
        self.observation = observation
        self.trial_cnt, self.timestep_cnt, self.neuron_cnt = observation.shape
 
    def plot_neuron_activity(self):
        """Plot the activity of all neurons for each trial."""
        fig, axes = plt.subplots(self.trial_cnt, 1, figsize=(10, 6 * self.trial_cnt))

        for trial_index in range(self.trial_cnt):
            subplot_title = f'Trial {trial_index + 1}'
            for neuron in range(self.neuron_cnt):
                axes[trial_index].plot(self.observation[trial_index, :, neuron], label=f'Neuron {neuron}')
            axes[trial_index].set_title(subplot_title)
            axes[trial_index].set_xlabel('Timepoints')
            axes[trial_index].set_ylabel('Activity')
            axes[trial_index].legend()
        plt.show()

    def plot_pca_explained_variance(self):
        """Plot the cumulative explained variance of PCA to determine optimal components."""
        data_reshaped = self.observation.reshape(-1, self.neuron_cnt)
        pca = PCA()
        pca.fit(data_reshaped)
        
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, self.neuron_cnt + 1), np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
        plt.axhline(y=0.9, color='r', linestyle='-', label='90% Variance threshold')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Explained Variance (Scree Plot)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_latent_dynamics(self, n_components: int = 3):
        """Reduce data to latent space using PCA and plot the dynamics."""
        data_reshaped = self.observation.reshape(-1, self.neuron_cnt)
        pca = PCA(n_components=n_components)
        latent = pca.fit_transform(data_reshaped)
        latent = latent.reshape(self.trial_cnt, self.timestep_cnt, -1)
        
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
        plt.tight_layout()
        plt.show()
        