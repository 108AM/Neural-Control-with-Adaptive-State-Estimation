import numpy as np
import matplotlib.pyplot as plt

class Illustrator:
    """
    A class to visualize neural activity data.
    Please finish the implementation and annotations of this class.
    """
    __slot__ = ['observation', 'trial_cnt', 'timestep_cnt', 'neuron_cnt']

    def __init__(self, observation: np.ndarray):
        """
        Initialize the Illustrator with the given observation data.
        Accepts a 3D numpy array of shape (Trials, Timepoints, Neurons) and stores it for visualization.
        Parameters:
            observation (np.ndarray): A 3D array containing neural activity data with dimensions (Trials, Timepoints, Neurons).
        """
        self.observation = observation
        self.trial_cnt, self.timestep_cnt, self.neuron_cnt = observation.shape
    
    def hello(self):
        print()
        
    # def plot_neuron_activity(self, trial_index: int):
    #     """Plot the activity of all neurons for a specific trial.
    #     Parameters:
    #         trial_index (int): The index of the trial to visualize.
    #     """
    #     if trial_index < 0 or trial_index >= self.trial_cnt:
    #         raise ValueError("Trial index out of range.")
        
    #     plt.figure(figsize=(10, 6))
    #     for neuron in range(self.neuron_cnt):
    #         plt.plot(self.observation[trial_index, :, neuron], label=f'Neuron {neuron}')
        
    #     plt.title(f'Neuron Activity for Trial {trial_index}')
    #     plt.xlabel('Timepoints')
    #     plt.ylabel('Activity')
    #     plt.legend()
    #     plt.show()
    