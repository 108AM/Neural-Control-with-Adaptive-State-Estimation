from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray


class InputSignal:
    """Immutable control input signal with numpy array protocol support.

    Represents a T-by-m array of control inputs (T timesteps, m channels).
    Follows the numpy array protocol (__array__) for seamless interoperability
    with functions expecting array-like inputs.
    """

    def __init__(
        self,
        T: int,
        m: int,
        array: NDArray,
        pattern_name: Optional[str] = None,
        dtype: type = np.float64,
    ) -> None:
        """Initialize an InputSignal.

        :param T: Number of timesteps.
        :param m: Number of input channels.
        :param array: Control input array of shape (T, m).
        :param pattern_name: Human-readable description of the pattern (e.g., 'pulse', 'composed').
        :param dtype: NumPy dtype for the array.
        :raises ValueError: If array shape does not match (T, m).
        """
        array = np.asarray(array, dtype=dtype)
        if array.shape != (T, m):
            raise ValueError(
                f"Array shape {array.shape} does not match expected shape ({T}, {m})"
            )
        self._signal = array
        self.T = T
        self.m = m
        self.pattern_name = pattern_name
        self.dtype = dtype

    def __array__(self) -> NDArray:
        """Return a copy of the signal array for numpy protocol."""
        return self._signal.copy()

    @property
    def shape(self) -> tuple[int, int]:
        """Return the shape of the signal array (T, m)."""
        return self._signal.shape

    def __getitem__(self, key):
        """Support indexing like a NumPy array."""
        return self._signal[key]

    def __repr__(self) -> str:
        """Return human-readable representation."""
        pattern = f", pattern='{self.pattern_name}'" if self.pattern_name else ""
        return f"InputSignal(T={self.T}, m={self.m}{pattern})"

    def to_array(self) -> NDArray:
        """Return a copy of the underlying array.

        :returns: Array of shape (T, m).
        """
        return self._signal.copy()

    @classmethod
    def zero(cls, T: int, m: int, dtype: type = np.float64) -> InputSignal:
        """Create all-zero input signal.

        :param T: Number of timesteps.
        :param m: Number of input channels.
        :param dtype: NumPy dtype.
        :returns: InputSignal with all zeros.
        """
        return cls(T, m, np.zeros((T, m), dtype=dtype), pattern_name="zero", dtype=dtype)

    @classmethod
    def random_gaussian(
        cls,
        T: int,
        m: int,
        scale: float = 1.0,
        rng: Optional[Generator] = None,
        dtype: type = np.float64,
    ) -> InputSignal:
        """Create IID Gaussian input signal.

        :param T: Number of timesteps.
        :param m: Number of input channels.
        :param scale: Standard deviation of the Gaussian distribution.
        :param rng: Optional seeded NumPy generator. Defaults to unseeded generator.
        :param dtype: NumPy dtype.
        :returns: InputSignal with Gaussian-distributed values.
        """
        rng = rng or np.random.default_rng()
        array = rng.normal(0.0, scale, (T, m)).astype(dtype)
        return cls(T, m, array, pattern_name="random_gaussian", dtype=dtype)

    @classmethod
    def pulse(
        cls,
        T: int,
        m: int,
        onset: int,
        duration: int = 1,
        channels: Optional[list[int]] = None,
        amplitude: float = 1.0,
        dtype: type = np.float64,
    ) -> InputSignal:
        """Create rectangular pulse on selected input channels.

        :param T: Number of timesteps.
        :param m: Number of input channels.
        :param onset: Timestep at which the pulse starts (0-indexed).
        :param duration: Number of active timesteps.
        :param channels: Which channels are active. Defaults to all channels.
        :param amplitude: Value during the pulse.
        :param dtype: NumPy dtype.
        :returns: InputSignal with pulse pattern.
        :raises ValueError: If onset or duration are invalid.
        """
        if onset < 0 or onset >= T:
            raise ValueError(f"onset={onset} must be in range [0, {T})")
        if duration < 1:
            raise ValueError(f"duration={duration} must be at least 1")
        if onset + duration > T:
            raise ValueError(f"Pulse extends beyond T: onset={onset}, duration={duration}, T={T}")

        array = np.zeros((T, m), dtype=dtype)
        cols = channels if channels is not None else list(range(m))
        array[onset : onset + duration, cols] = amplitude
        return cls(T, m, array, pattern_name="pulse", dtype=dtype)

    @classmethod
    def oscillatory(
        cls,
        T: int,
        m: int,
        frequency: float,
        amplitude: float = 1.0,
        phase: float = 0.0,
        channels: Optional[list[int]] = None,
        dtype: type = np.float64,
    ) -> InputSignal:
        """Create sinusoidal input on selected channels.

        :param T: Number of timesteps.
        :param m: Number of input channels.
        :param frequency: Frequency in cycles per timestep (e.g., 0.05 gives 1 cycle per 20 steps).
        :param amplitude: Peak amplitude.
        :param phase: Initial phase offset in radians.
        :param channels: Which channels to activate. Defaults to all channels.
        :param dtype: NumPy dtype.
        :returns: InputSignal with sinusoidal pattern.
        """
        t = np.arange(T, dtype=dtype)
        signal = amplitude * np.sin(2 * np.pi * frequency * t + phase).astype(dtype)

        array = np.zeros((T, m), dtype=dtype)
        cols = channels if channels is not None else list(range(m))
        array[:, cols] = signal[:, np.newaxis]
        return cls(T, m, array, pattern_name="oscillatory", dtype=dtype)

    @classmethod
    def single_channel(
        cls,
        T: int,
        m: int,
        channel: int,
        pattern: NDArray,
        dtype: type = np.float64,
    ) -> InputSignal:
        """Apply a 1-D pattern to one input channel; all others are zero.

        :param T: Number of timesteps.
        :param m: Number of input channels.
        :param channel: Index of the channel to activate.
        :param pattern: 1-D array of length T.
        :param dtype: NumPy dtype.
        :returns: InputSignal with pattern on specified channel.
        :raises ValueError: If pattern length does not match T.
        """
        pattern = np.asarray(pattern, dtype=dtype)
        if len(pattern) != T:
            raise ValueError(f"Pattern length {len(pattern)} does not match T={T}")

        array = np.zeros((T, m), dtype=dtype)
        array[:, channel] = pattern
        return cls(T, m, array, pattern_name="single_channel", dtype=dtype)

    @classmethod
    def ramp(
        cls,
        T: int,
        m: int,
        channel: int,
        start: float = 0.0,
        end: float = 1.0,
        dtype: type = np.float64,
    ) -> InputSignal:
        """Create linearly ramping input on one channel, all others zero.

        :param T: Number of timesteps.
        :param m: Number of input channels.
        :param channel: Index of the channel to activate.
        :param start: Value at t=0.
        :param end: Value at t=T-1.
        :param dtype: NumPy dtype.
        :returns: InputSignal with ramp pattern.
        """
        array = np.zeros((T, m), dtype=dtype)
        array[:, channel] = np.linspace(start, end, T, dtype=dtype)
        return cls(T, m, array, pattern_name="ramp", dtype=dtype)


class InputBuilder:
    """Fluent interface for composing multiple input patterns.

    Enables method chaining to combine different patterns on different channels.
    Call build() to produce an immutable InputSignal.

    Example:
        u = (InputBuilder(T=120, m=3)
             .pulse(onset=30, duration=2, channels=[0], amplitude=1.0)
             .oscillatory(frequency=0.05, channels=[1, 2])
             .build())
    """

    def __init__(self, T: int, m: int, dtype: type = np.float64) -> None:
        """Initialize a builder.

        :param T: Number of timesteps.
        :param m: Number of input channels.
        :param dtype: NumPy dtype for the composed signal.
        """
        self.T = T
        self.m = m
        self.dtype = dtype
        self._signal = np.zeros((T, m), dtype=dtype)

    def zero(self) -> InputBuilder:
        """Add zero pattern (no-op, array is already initialized to zero).

        :returns: Self for method chaining.
        """
        return self

    def pulse(
        self,
        onset: int,
        duration: int = 1,
        channels: Optional[list[int]] = None,
        amplitude: float = 1.0,
    ) -> InputBuilder:
        """Superimpose a rectangular pulse on selected channels.

        :param onset: Timestep at which the pulse starts (0-indexed).
        :param duration: Number of active timesteps.
        :param channels: Which channels to activate. Defaults to all channels.
        :param amplitude: Value during the pulse.
        :returns: Self for method chaining.
        :raises ValueError: If onset or duration are invalid.
        """
        if onset < 0 or onset >= self.T:
            raise ValueError(f"onset={onset} must be in range [0, {self.T})")
        if duration < 1:
            raise ValueError(f"duration={duration} must be at least 1")
        if onset + duration > self.T:
            raise ValueError(f"Pulse extends beyond T: onset={onset}, duration={duration}, T={self.T}")

        cols = channels if channels is not None else list(range(self.m))
        self._signal[onset : onset + duration, cols] += amplitude
        return self

    def oscillatory(
        self,
        frequency: float,
        channels: Optional[list[int]] = None,
        amplitude: float = 1.0,
        phase: float = 0.0,
    ) -> InputBuilder:
        """Superimpose a sinusoidal pattern on selected channels.

        :param frequency: Frequency in cycles per timestep.
        :param channels: Which channels to activate. Defaults to all channels.
        :param amplitude: Peak amplitude.
        :param phase: Initial phase offset in radians.
        :returns: Self for method chaining.
        """
        t = np.arange(self.T, dtype=self.dtype)
        signal = amplitude * np.sin(2 * np.pi * frequency * t + phase).astype(self.dtype)
        cols = channels if channels is not None else list(range(self.m))
        self._signal[:, cols] += signal[:, np.newaxis]
        return self

    def single_channel(
        self,
        channel: int,
        pattern: NDArray,
    ) -> InputBuilder:
        """Superimpose a 1-D pattern on one channel.

        :param channel: Index of the channel.
        :param pattern: 1-D array of length T.
        :returns: Self for method chaining.
        :raises ValueError: If pattern length does not match T.
        """
        pattern = np.asarray(pattern, dtype=self.dtype)
        if len(pattern) != self.T:
            raise ValueError(f"Pattern length {len(pattern)} does not match T={self.T}")
        self._signal[:, channel] += pattern
        return self

    def ramp(
        self,
        channel: int,
        start: float = 0.0,
        end: float = 1.0,
    ) -> InputBuilder:
        """Superimpose a linear ramp on one channel.

        :param channel: Index of the channel.
        :param start: Value at t=0.
        :param end: Value at t=T-1.
        :returns: Self for method chaining.
        """
        ramp_values = np.linspace(start, end, self.T, dtype=self.dtype)
        self._signal[:, channel] += ramp_values
        return self

    def random_gaussian(
        self,
        channels: Optional[list[int]] = None,
        scale: float = 1.0,
        rng: Optional[Generator] = None,
    ) -> InputBuilder:
        """Superimpose Gaussian noise on selected channels.

        :param channels: Which channels to affect. Defaults to all channels.
        :param scale: Standard deviation.
        :param rng: Optional seeded generator.
        :returns: Self for method chaining.
        """
        rng = rng or np.random.default_rng()
        cols = channels if channels is not None else list(range(self.m))
        noise = rng.normal(0.0, scale, (self.T, len(cols))).astype(self.dtype)
        self._signal[:, cols] += noise
        return self

    def build(self) -> InputSignal:
        """Create the final immutable InputSignal from composed patterns.

        :returns: InputSignal with superimposed patterns.
        """
        return InputSignal(
            T=self.T,
            m=self.m,
            array=self._signal.copy(),
            pattern_name="composed",
            dtype=self.dtype,
        )
