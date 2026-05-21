from __future__ import annotations

import numpy as np
from numpy.random import Generator


def zero(T: int, m: int) -> np.ndarray:
    """All-zero input of shape ``(T, m)``."""
    return np.zeros((T, m))


def random_gaussian(
    T: int, m: int, scale: float = 1.0, rng: Generator | None = None
) -> np.ndarray:
    """IID Gaussian input of shape ``(T, m)`` with standard deviation *scale*.

    :param rng: Optional seeded generator. Defaults to a fresh unseeded one.
    """
    rng = rng or np.random.default_rng()
    return rng.normal(0.0, scale, (T, m))


def pulse(
    T: int,
    m: int,
    onset: int,
    duration: int = 1,
    channels: list[int] | None = None,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Rectangular pulse on selected input channels, shape ``(T, m)``.

    :param onset: Timestep at which the pulse starts (0-indexed).
    :param duration: Number of active timesteps.
    :param channels: Which channels are active. Defaults to all channels.
    :param amplitude: Value during the pulse.
    """
    u = np.zeros((T, m))
    cols = channels if channels is not None else list(range(m))
    u[onset : onset + duration, cols] = amplitude
    return u


def oscillatory(
    T: int,
    m: int,
    freq: float,
    amplitude: float = 1.0,
    phase: float = 0.0,
) -> np.ndarray:
    """Sinusoidal input on all channels, shape ``(T, m)``.

    :param freq: Frequency in cycles per timestep (e.g. ``0.05`` → 1 cycle per 20 steps).
    :param amplitude: Peak amplitude.
    :param phase: Initial phase offset in radians.
    """
    t = np.arange(T)
    signal = amplitude * np.sin(2 * np.pi * freq * t + phase)
    return np.tile(signal[:, None], (1, m))


def single_channel(
    T: int,
    m: int,
    channel: int,
    pattern: np.ndarray,
) -> np.ndarray:
    """Apply a 1-D pattern to one input channel; all others are zero.

    :param pattern: 1-D array of length *T*.
    """
    u = np.zeros((T, m))
    u[:, channel] = np.asarray(pattern)
    return u


def ramp(
    T: int,
    m: int,
    channel: int,
    start: float = 0.0,
    end: float = 1.0,
) -> np.ndarray:
    """Linearly ramping input on one channel, all others zero.

    :param start: Value at ``t=0``.
    :param end: Value at ``t=T-1``.
    """
    u = np.zeros((T, m))
    u[:, channel] = np.linspace(start, end, T)
    return u
