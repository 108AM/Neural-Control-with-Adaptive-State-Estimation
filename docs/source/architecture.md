# Architecture

## Core classes

### {class}`~l4b.simulator.Simulator`

`Simulator` implements a discrete-time linear-Gaussian state-space model:

$$
x_{t+1} = A x_t + B u_t + w_t, \quad w_t \sim \mathcal{N}(0, Q)
$$

$$
y_t = C x_t + v_t, \quad v_t \sim \mathcal{N}(0, R)
$$

It is initialised from a plain `dict` that specifies the system matrices
(`A`, `B`, `C`, `Q`, `R`) and dimensions (`state_dim`, `input_dim`, `obs_dim`).
Calling {meth}`~l4b.simulator.Simulator.run` returns raw NumPy arrays of states
and observations across multiple trials. Wrap the observations in a
{class}`~l4b.dataset.Dataset` to pass them to the analysis layer.

`Simulator` also exposes the finite-horizon
{meth}`~l4b.simulator.Simulator.controllability_gramian` and
{meth}`~l4b.simulator.Simulator.observability_gramian` for analysing system
properties independently of a specific run.

### {class}`~l4b.dataset.Dataset`

`Dataset` is a frozen dataclass that wraps a 3-D observations array of shape
`(n_trials, n_timesteps, n_neurons)`. Its immutability means the same dataset
can be shared freely across multiple analyses without risk of mutation.

### {class}`~l4b.illustrator.Illustrator`

`Illustrator` is a thin facade that owns a `Dataset` and drives plotting
pipelines. Each `plot_*` method internally constructs the corresponding
{class}`~l4b.stats.analysis.Analysis` subclass, calls `.result`, and renders a
Matplotlib `Figure`.

It can be constructed either directly (`Illustrator(dataset)`) or via the
convenience class method `Illustrator.from_array(arr)`.

The `summary()` and `plot_all()` methods generate a full suite of plots in
one call.

## `stats` subpackage: analyses

All analyses share a common base class, {class}`~l4b.stats.analysis.Analysis`,
which implements lazy, cached computation: the first access of `.result`
calls the abstract `_compute()` method and stores the value via
`functools.cached_property`; subsequent accesses return the cached value at no
extra cost.

### Summary table

| Class                                                  | Result type                                                  | What it computes                                                |
| ------------------------------------------------------ | ------------------------------------------------------------ | --------------------------------------------------------------- |
| {class}`~l4b.stats.mean.Mean`                          | {class}`~l4b.stats.mean.MeanResult`                          | Trial-averaged activity and standard error per neuron           |
| {class}`~l4b.stats.correlation.Correlation`            | `NDArray`                                                    | Pearson correlation matrix across neurons for a single trial    |
| {class}`~l4b.stats.pca.PCA`                            | {class}`~l4b.stats.pca.PCAResult`                            | Principal components of the trial-averaged activity             |
| {class}`~l4b.stats.autocorrelation.Autocorrelation`    | {class}`~l4b.stats.autocorrelation.AutocorrelationResult`    | Per-neuron autocorrelation function averaged across trials      |
| {class}`~l4b.stats.power_spectrum.PowerSpectrum`       | {class}`~l4b.stats.power_spectrum.PowerSpectrumResult`       | Welch power spectrum per neuron averaged across trials          |
| {class}`~l4b.stats.coherence.Coherence`                | {class}`~l4b.stats.coherence.CoherenceResult`                | Magnitude-squared coherence for every neuron pair across trials |
| {class}`~l4b.stats.cross_correlation.CrossCorrelation` | {class}`~l4b.stats.cross_correlation.CrossCorrelationResult` | Normalised cross-correlation between two neurons across trials  |

The helper function {func}`~l4b.stats.mean.population_centre` computes the
population-level centre of mass from a `MeanResult`.

## Input signal generation

Control signals are generated using {class}`~l4b.inputs.InputSignal` (for individual
patterns) and {class}`~l4b.inputs.InputBuilder` (for composing multiple patterns on
different channels). See [Composing Input Signals](composing_inputs.md) for detailed usage
and design rationale.

## A typical workflow

```python
import numpy as np
import l4b

# 1. Define the system
model = {
    "state_dim": 4, "input_dim": 2, "obs_dim": 6,
    "A": ..., "B": ..., "C": ..., "Q": ..., "R": ...,
}
sim = l4b.Simulator(model)

# 2. Generate control inputs and run the simulation
u = l4b.InputSignal.pulse(T=200, m=2, onset=50, duration=10)
_, obs = sim.run(initial_state=np.zeros(4), control_inputs=u, time_steps=200, trials=50)

# 3. Wrap observations in a Dataset
dataset = l4b.Dataset(obs)

# 4. Plot
illustrator = l4b.Illustrator(dataset)
illustrator.plot_all()
```
