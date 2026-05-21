# Composing Input Signals

The {class}`~l4b.inputs.InputSignal` and {class}`~l4b.inputs.InputBuilder` classes provide ergonomic methods for generating and combining control input patterns.

## InputSignal: Individual patterns

`InputSignal` is an immutable object that represents a T-by-m control signal. Create patterns using class methods:

```python
from l4b import InputSignal

u_zero = InputSignal.zero(T=120, m=3)
u_pulse = InputSignal.pulse(T=120, m=3, onset=30, duration=2, amplitude=1.0)
u_osc = InputSignal.oscillatory(T=120, m=3, frequency=0.05)
u_ramp = InputSignal.ramp(T=120, m=3, channel=1, start=0.0, end=1.0)
```

Available patterns are listed in the table below. Each method accepts optional `channels` to apply the pattern to specific inputs (defaults to all channels). The resulting `InputSignal` integrates transparently with {meth}`~l4b.simulator.Simulator.run`:

```python
from l4b import Simulator

sim = Simulator(model)
states, obs = sim.run(initial_state=x0, control_inputs=u_pulse, time_steps=120, trials=10)
```

`InputSignal` implements the NumPy array protocol (`__array__`, `__getitem__`, `.shape`), so it works anywhere array-like inputs are expected.

| Class method                                    | Description                                            |
| ----------------------------------------------- | ------------------------------------------------------ |
| {meth}`~l4b.inputs.InputSignal.zero`            | All-zero inputs (baseline)                             |
| {meth}`~l4b.inputs.InputSignal.random_gaussian` | IID Gaussian noise with standard deviation `scale`     |
| {meth}`~l4b.inputs.InputSignal.pulse`           | Rectangular pulse with onset, duration, and amplitude  |
| {meth}`~l4b.inputs.InputSignal.oscillatory`     | Sinusoidal signal with frequency, amplitude, and phase |
| {meth}`~l4b.inputs.InputSignal.single_channel`  | Arbitrary 1-D pattern on one channel                   |
| {meth}`~l4b.inputs.InputSignal.ramp`            | Linear ramp from `start` to `end` on one channel       |

## InputBuilder: Composing patterns

When you need multiple patterns on different channels, use {class}`~l4b.inputs.InputBuilder` with method chaining:

```python
from l4b import InputBuilder

u = (InputBuilder(T=120, m=3)
     .pulse(onset=30, duration=2, channels=[0], amplitude=1.0)
     .oscillatory(frequency=0.05, channels=[1, 2])
     .build())
```

Each builder method superimposes a pattern onto the signal and returns `self` for chaining. Call `.build()` to produce the final immutable `InputSignal`.

This design separates concerns: `InputSignal` is an immutable value, while `InputBuilder` is a mutable factory for composition. The pattern is ergonomic for systematic exploration where each channel receives different driving patterns or the same pattern at different amplitudes:

```python
u = (InputBuilder(T=100, m=2)
     .pulse(onset=20, duration=5, channels=[0, 1], amplitude=1.0)
     .pulse(onset=20, duration=5, channels=[1], amplitude=0.5))
     .build()
```

Channel 0 has amplitude 1.0, channel 1 has 1.0 + 0.5 = 1.5 from the superposition.

## Design rationale

`InputSignal` is immutable to prevent accidental modification. It stores the underlying NumPy array but exposes only read-only access via array protocol methods. This follows the immutability pattern used throughout l4b (frozen `Dataset`, NamedTuple results from analyses).

`InputBuilder` is mutable during construction but returns immutable `InputSignal` objects. This avoids creating intermediate `InputSignal` objects for every step of a composition chain, keeping both the interface clean and memory usage efficient.

Full NumPy compatibility (`__array__`, `__getitem__`, `.shape`) ensures seamless integration with existing code. You never need to explicitly convert to NumPy arrays; indexing and array operations work directly on `InputSignal` objects.

## Type safety and reproducibility

All generators accept a `dtype` parameter (default `np.float64`) for numerical precision control. Generators that use randomness accept an optional seeded `rng` (NumPy Generator) for reproducibility:

```python
from numpy.random import default_rng

rng = default_rng(seed=42)
u = InputSignal.random_gaussian(T=100, m=2, scale=1.0, rng=rng)
```
