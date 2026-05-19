# Illustrator

```{eval-rst}
.. py:currentmodule:: illustrator

.. autoclass:: Illustrator
   :members:
   :show-inheritance:
   :member-order: bysource
```

## Class diagram

```{mermaid}
classDiagram
    class Simulator {
        +int state_dim
        +int input_dim
        +int obs_dim
        +ndarray A
        +ndarray B
        +ndarray C
        +ndarray Q
        +ndarray R
        -Generator generator
        +__init__(model: dict) None
        +run(initial_state: ndarray, control_inputs: ndarray, time_steps: int, trials: int) tuple
        +test() None
    }

    class Dataset {
        <<dataclass frozen>>
        -NDArray _observations
        +observations() NDArray
        +shape() tuple
        +n_trials() int
        +n_timesteps() int
        +n_neurons() int
        +__init__(observations: ArrayLike) None
        +__repr__() str
    }

    class Analysis {
        <<abstract>>
        #Dataset _dataset
        +result() Any
        +__init__(dataset: Dataset) None
        #_compute()* Any
    }

    class Mean {
        +result() MeanResult
        #_compute() MeanResult
    }

    class Correlation {
        -int trial
        +result() NDArray
        +__init__(dataset: Dataset, trial: int) None
        #_compute() NDArray
    }

    class PCA {
        -int n_components
        +result() PCAResult
        +__init__(dataset: Dataset, n_components: int) None
        #_compute() PCAResult
    }

    class Autocorrelation {
        -int max_lag
        +result() AutocorrelationResult
        +__init__(dataset: Dataset, max_lag: int) None
        #_compute() AutocorrelationResult
    }

    class PowerSpectrum {
        -float fs
        -int nperseg
        +result() PowerSpectrumResult
        +__init__(dataset: Dataset, fs: float, nperseg: int) None
        #_compute() PowerSpectrumResult
    }

    class Illustrator {
        -Dataset _dataset
        +__init__(dataset: Dataset) None
        +from_array(arr: ArrayLike)$ Illustrator
        +summary() None
        +plot_neurons(...) Figure
        +plot_mean_and_std(...) Figure
        +plot_heatmap(...) Figure
        +plot_neuron_heatmap(...) Figure
        +plot_correlation(...) Figure
        +plot_pca(...) Figure
        +plot_variance_explained(...) Figure
        +plot_autocorrelation(...) Figure
        +plot_power_spectrum(...) Figure
        +plot_all() None
    }

    Analysis <|-- Mean : extends
    Analysis <|-- Correlation : extends
    Analysis <|-- PCA : extends
    Analysis <|-- Autocorrelation : extends
    Analysis <|-- PowerSpectrum : extends
    Analysis o-- Dataset : has
    Illustrator o-- Dataset : has
    Illustrator ..> Mean : uses
    Illustrator ..> Correlation : uses
    Illustrator ..> PCA : uses
    Illustrator ..> Autocorrelation : uses
    Illustrator ..> PowerSpectrum : uses
```
