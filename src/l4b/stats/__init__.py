"""Statistical analysis modules for neural recording datasets"""

from l4b.stats.analysis import Analysis
from l4b.stats.autocorrelation import Autocorrelation, AutocorrelationResult
from l4b.stats.coherence import Coherence, CoherenceResult
from l4b.stats.correlation import Correlation
from l4b.stats.cross_correlation import CrossCorrelation, CrossCorrelationResult
from l4b.stats.mean import Mean, MeanResult, population_centre
from l4b.stats.pca import PCA
from l4b.stats.power_spectrum import PowerSpectrum

__all__ = [
    "Analysis",
    "Autocorrelation",
    "AutocorrelationResult",
    "Coherence",
    "CoherenceResult",
    "Correlation",
    "CrossCorrelation",
    "CrossCorrelationResult",
    "Mean",
    "MeanResult",
    "PCA",
    "PowerSpectrum",
    "population_centre",
]
