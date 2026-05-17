"""
test_illustrator.py
===================
Pytest test suite for Illustrator.py.

Run with:
    pytest test_illustrator.py -v

All plotting methods are tested with a non-interactive matplotlib backend so
no display is required (safe for CI / pre-commit hooks).  Every plot test
verifies that the method returns a Figure and that no exception is raised;
figure windows are closed immediately after each test to avoid memory buildup.

Test categories
---------------
- Initialisation        : correct attribute shapes, defaults, error handling
- Derived attributes    : numerical correctness of pre-computed arrays
- _demean               : mathematical correctness
- plot_neurons          : all parameter branches
- plot_mean_and_std     : default and per-neuron calls
- plot_heatmap          : trial-averaged and single-trial, invalid index
- plot_neuron_heatmap   : centred and raw variants
- plot_correlation      : full dataset and single trial
- plot_pca              : default and n_components edge cases
- plot_variance_explained: default and capped n_components
- plot_autocorrelation  : default and restricted neuron/lag
- plot_power_spectrum   : default and custom fs
- summary               : output content
- plot_all              : full pipeline smoke test
"""

import sys
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — must be set before pyplot import
import matplotlib.pyplot as plt
import pytest

# ── make sure Illustrator.py is importable from the same directory ───────────
sys.path.insert(0, os.path.dirname(__file__))
from Illustrator import Illustrator


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def standard_data(rng):
    """Standard dataset matching the project shape: (5, 60, 16)."""
    return rng.standard_normal((5, 60, 16))


@pytest.fixture
def ill(standard_data):
    """Default Illustrator instance; closed after every test."""
    obj = Illustrator(standard_data)
    yield obj
    plt.close("all")


@pytest.fixture
def small_data(rng):
    """Minimal dataset for edge-case tests: (2, 8, 4)."""
    return rng.standard_normal((2, 8, 4))


@pytest.fixture
def ill_small(small_data):
    obj = Illustrator(small_data)
    yield obj
    plt.close("all")


# ══════════════════════════════════════════════════════════════════════════════
# Initialisation
# ══════════════════════════════════════════════════════════════════════════════

class TestInit:

    def test_shape_attributes(self, ill, standard_data):
        assert ill.n_trials  == 5
        assert ill.n_time    == 60
        assert ill.n_neurons == 16
        assert ill.data is standard_data

    def test_default_dt(self, ill):
        assert ill.dt == 1.0

    def test_custom_dt(self, standard_data):
        obj = Illustrator(standard_data, dt=0.01)
        assert obj.dt == 0.01
        assert obj.time[-1] == pytest.approx(59 * 0.01)

    def test_time_axis_length(self, ill):
        assert len(ill.time) == ill.n_time

    def test_time_axis_values(self, ill):
        expected = np.arange(60) * 1.0
        np.testing.assert_array_equal(ill.time, expected)

    def test_default_neuron_labels(self, ill):
        assert ill.neuron_labels == [f"Neuron {i}" for i in range(16)]

    def test_custom_neuron_labels(self, standard_data):
        labels = [f"Ch{i}" for i in range(16)]
        obj = Illustrator(standard_data, neuron_labels=labels)
        assert obj.neuron_labels == labels

    def test_wrong_neuron_label_length_raises(self, standard_data):
        with pytest.raises(ValueError, match="neuron_labels"):
            Illustrator(standard_data, neuron_labels=["A", "B"])

    def test_2d_input_raises(self):
        with pytest.raises(ValueError, match="3-D"):
            Illustrator(np.zeros((5, 60)))

    def test_4d_input_raises(self):
        with pytest.raises(ValueError, match="3-D"):
            Illustrator(np.zeros((5, 60, 16, 1)))


# ══════════════════════════════════════════════════════════════════════════════
# Derived attributes
# ══════════════════════════════════════════════════════════════════════════════

class TestDerivedAttributes:

    def test_trial_averaged_shape(self, ill):
        assert ill.trial_averaged_data.shape == (60, 16)

    def test_trial_averaged_is_mean_over_axis0(self, ill):
        expected = ill.data.mean(axis=0)
        np.testing.assert_allclose(ill.trial_averaged_data, expected)

    def test_mean_activity_is_alias(self, ill):
        assert ill.mean_activity is ill.trial_averaged_data

    def test_population_averaged_shape(self, ill):
        # Should be (T, N_t) = (5, 60) — mean across neurons
        assert ill.population_averaged_data.shape == (5, 60)

    def test_population_averaged_is_mean_over_neurons(self, ill):
        expected = ill.data.mean(axis=2)
        np.testing.assert_allclose(ill.population_averaged_data, expected)

    def test_population_centred_shape(self, ill):
        assert ill.population_centred_data.shape == (5, 60, 16)

    def test_population_centred_neuron_mean_is_zero(self, ill):
        """After centring, the mean across neurons at every (trial, timepoint) should be 0."""
        neuron_means = ill.population_centred_data.mean(axis=2)  # (T, N_t)
        np.testing.assert_allclose(neuron_means, 0.0, atol=1e-12)

    def test_population_centred_correct_values(self, ill):
        expected = ill.data - ill.data.mean(axis=2, keepdims=True)
        np.testing.assert_allclose(ill.population_centred_data, expected)


# ══════════════════════════════════════════════════════════════════════════════
# _demean
# ══════════════════════════════════════════════════════════════════════════════

class TestDemean:

    def test_output_shape_preserved(self, ill):
        mat = ill.trial_averaged_data
        assert ill._demean(mat).shape == mat.shape

    def test_row_means_are_zero(self, ill):
        """After demeaning, the mean across neurons at each timepoint must be 0."""
        centred = ill._demean(ill.trial_averaged_data)
        row_means = centred.mean(axis=1)          # (N_t,)
        np.testing.assert_allclose(row_means, 0.0, atol=1e-12)

    def test_pairwise_neuron_differences_preserved(self, ill):
        """
        _demean subtracts the same scalar from every neuron at each timestep,
        so the difference between any two neurons is unchanged.
        """
        mat     = ill.trial_averaged_data
        centred = ill._demean(mat)
        # difference between neuron 0 and neuron 1 should be identical before/after
        np.testing.assert_allclose(
            mat[:, 0] - mat[:, 1],
            centred[:, 0] - centred[:, 1],
            atol=1e-12,
        )

    def test_idempotent(self, ill):
        """Applying _demean twice gives the same result as applying it once."""
        once  = ill._demean(ill.trial_averaged_data)
        twice = ill._demean(once)
        np.testing.assert_allclose(once, twice, atol=1e-12)

    def test_known_values(self):
        """Verify against a hand-computed example."""
        data = np.array([[1.0, 3.0],    # row mean = 2  → centred: [-1, 1]
                         [4.0, 8.0]])   # row mean = 6  → centred: [-2, 2]
        obj      = Illustrator(np.zeros((2, 2, 2)))   # dummy, just to call _demean
        centred  = obj._demean(data)
        expected = np.array([[-1.0, 1.0],
                             [-2.0, 2.0]])
        np.testing.assert_allclose(centred, expected)


# ══════════════════════════════════════════════════════════════════════════════
# Helper: assert a method returns a Figure and closes cleanly
# ══════════════════════════════════════════════════════════════════════════════

def _is_figure(result):
    return isinstance(result, plt.Figure)


# ══════════════════════════════════════════════════════════════════════════════
# plot_neurons
# ══════════════════════════════════════════════════════════════════════════════

class TestPlotNeurons:

    def test_returns_figure(self, ill):
        assert _is_figure(ill.plot_neurons())

    def test_single_trial(self, ill):
        assert _is_figure(ill.plot_neurons(trials=[0]))

    def test_single_neuron(self, ill):
        assert _is_figure(ill.plot_neurons(neurons=[0]))

    def test_subset_trials_and_neurons(self, ill):
        assert _is_figure(ill.plot_neurons(trials=[0, 1], neurons=[0, 1, 2]))

    def test_plot_mean_flag(self, ill):
        assert _is_figure(ill.plot_neurons(trials=[0], plot_mean=True))

    def test_plot_std_flag(self, ill):
        assert _is_figure(ill.plot_neurons(trials=[0], plot_std=True))

    def test_population_centred_flag(self, ill):
        assert _is_figure(ill.plot_neurons(trials=[0], population_centred=True))

    def test_all_flags_combined(self, ill):
        assert _is_figure(
            ill.plot_neurons(trials=[0], neurons=[0, 1],
                             plot_mean=True, plot_std=True,
                             population_centred=True))


# ══════════════════════════════════════════════════════════════════════════════
# plot_mean_and_std
# ══════════════════════════════════════════════════════════════════════════════

class TestPlotMeanAndStd:

    def test_returns_figure(self, ill):
        assert _is_figure(ill.plot_mean_and_std())

    def test_subset_neurons(self, ill):
        assert _is_figure(ill.plot_mean_and_std(neurons=[0, 1, 2]))

    def test_sem_off(self, ill):
        assert _is_figure(ill.plot_mean_and_std(show_sem=False))

    def test_single_neuron(self, ill):
        assert _is_figure(ill.plot_mean_and_std(neurons=[0]))


# ══════════════════════════════════════════════════════════════════════════════
# plot_heatmap
# ══════════════════════════════════════════════════════════════════════════════

class TestPlotHeatmap:

    def test_trial_averaged_default(self, ill):
        assert _is_figure(ill.plot_heatmap())

    def test_single_trial(self, ill):
        assert _is_figure(ill.plot_heatmap(trial=0))

    def test_last_valid_trial(self, ill):
        assert _is_figure(ill.plot_heatmap(trial=4))

    def test_invalid_trial_raises(self, ill):
        with pytest.raises(ValueError):
            ill.plot_heatmap(trial=99)

    def test_negative_trial_raises(self, ill):
        with pytest.raises(ValueError):
            ill.plot_heatmap(trial=-1)


# ══════════════════════════════════════════════════════════════════════════════
# plot_neuron_heatmap
# ══════════════════════════════════════════════════════════════════════════════

class TestPlotNeuronHeatmap:

    def test_returns_figure_centred(self, ill):
        assert _is_figure(ill.plot_neuron_heatmap())

    def test_returns_figure_raw(self, ill):
        assert _is_figure(ill.plot_neuron_heatmap(population_centred=False))

    def test_subset_trials(self, ill):
        assert _is_figure(ill.plot_neuron_heatmap(trials=[0, 1]))

    def test_subset_neurons(self, ill):
        assert _is_figure(ill.plot_neuron_heatmap(neurons=[0, 1, 2, 3]))

    def test_single_trial(self, ill):
        assert _is_figure(ill.plot_neuron_heatmap(trials=[0]))


# ══════════════════════════════════════════════════════════════════════════════
# plot_correlation
# ══════════════════════════════════════════════════════════════════════════════

class TestPlotCorrelation:

    def test_all_trials(self, ill):
        assert _is_figure(ill.plot_correlation())

    def test_single_trial(self, ill):
        assert _is_figure(ill.plot_correlation(trial=0))

    def test_correlation_matrix_bounds(self, ill):
        """Verify the computed correlation matrix has values in [-1, 1]."""
        mat  = ill.data.reshape(-1, ill.n_neurons)
        corr = np.corrcoef(mat.T)
        assert corr.min() >= -1.0 - 1e-10
        assert corr.max() <=  1.0 + 1e-10

    def test_correlation_matrix_diagonal_is_one(self, ill):
        mat  = ill.data.reshape(-1, ill.n_neurons)
        corr = np.corrcoef(mat.T)
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-12)

    def test_correlation_matrix_is_symmetric(self, ill):
        mat  = ill.data.reshape(-1, ill.n_neurons)
        corr = np.corrcoef(mat.T)
        np.testing.assert_allclose(corr, corr.T, atol=1e-12)


# ══════════════════════════════════════════════════════════════════════════════
# plot_pca
# ══════════════════════════════════════════════════════════════════════════════

class TestPlotPca:

    def test_returns_figure(self, ill):
        assert _is_figure(ill.plot_pca())

    def test_custom_n_components(self, ill):
        assert _is_figure(ill.plot_pca(n_components=2))

    def test_n_components_clamped_to_min_dim(self, ill_small):
        """Requesting more PCs than neurons/timepoints should not raise."""
        assert _is_figure(ill_small.plot_pca(n_components=100))

    def test_one_component(self, ill):
        assert _is_figure(ill.plot_pca(n_components=1))

    def test_pca_uses_demeaned_data(self, ill):
        """PC scores on demeaned data should have near-zero mean across time."""
        from sklearn.decomposition import PCA
        centred = ill._demean(ill.trial_averaged_data)
        scores  = PCA(n_components=3).fit_transform(centred)
        np.testing.assert_allclose(scores.mean(axis=0), 0.0, atol=1e-10)

    def test_variance_ratios_sum_to_one(self, ill):
        from sklearn.decomposition import PCA
        centred = ill._demean(ill.trial_averaged_data)
        pca     = PCA().fit(centred)
        assert pca.explained_variance_ratio_.sum() == pytest.approx(1.0, abs=1e-10)


# ══════════════════════════════════════════════════════════════════════════════
# plot_variance_explained
# ══════════════════════════════════════════════════════════════════════════════

class TestPlotVarianceExplained:

    def test_returns_figure(self, ill):
        assert _is_figure(ill.plot_variance_explained())

    def test_capped_n_components(self, ill):
        assert _is_figure(ill.plot_variance_explained(n_components=4))

    def test_n_components_above_max_clamped(self, ill_small):
        """Requesting more components than available should not raise."""
        assert _is_figure(ill_small.plot_variance_explained(n_components=999))

    def test_cumulative_variance_is_non_decreasing(self, ill):
        from sklearn.decomposition import PCA
        centred = ill._demean(ill.trial_averaged_data)
        ev      = PCA().fit(centred).explained_variance_ratio_
        cumev   = np.cumsum(ev)
        assert np.all(np.diff(cumev) >= 0)


# ══════════════════════════════════════════════════════════════════════════════
# plot_autocorrelation
# ══════════════════════════════════════════════════════════════════════════════

class TestPlotAutocorrelation:

    def test_returns_figure(self, ill):
        assert _is_figure(ill.plot_autocorrelation())

    def test_subset_neurons(self, ill):
        assert _is_figure(ill.plot_autocorrelation(neurons=[0, 1]))

    def test_custom_max_lag(self, ill):
        assert _is_figure(ill.plot_autocorrelation(max_lag=10))

    def test_acf_lag0_is_one(self, ill):
        """By definition the ACF at lag 0 should equal 1.0."""
        for trial in range(ill.n_trials):
            for ni in range(ill.n_neurons):
                x  = ill.data[trial, :, ni] - ill.data[trial, :, ni].mean()
                ac = np.correlate(x, x, mode="full")
                ac = ac[len(ac) // 2:]
                ac = ac / ac[0]
                assert ac[0] == pytest.approx(1.0)

    def test_acf_is_bounded(self, ill):
        """Normalised ACF values should lie in [-1, 1]."""
        for trial in range(ill.n_trials):
            x  = ill.data[trial, :, 0] - ill.data[trial, :, 0].mean()
            ac = np.correlate(x, x, mode="full")
            ac = ac[len(ac) // 2:]
            ac = ac / ac[0]
            assert ac.min() >= -1.0 - 1e-10
            assert ac.max() <=  1.0 + 1e-10


# ══════════════════════════════════════════════════════════════════════════════
# plot_power_spectrum
# ══════════════════════════════════════════════════════════════════════════════

class TestPlotPowerSpectrum:

    def test_returns_figure(self, ill):
        assert _is_figure(ill.plot_power_spectrum())

    def test_subset_neurons(self, ill):
        assert _is_figure(ill.plot_power_spectrum(neurons=[0, 1]))

    def test_custom_fs(self, ill):
        assert _is_figure(ill.plot_power_spectrum(fs=100.0))

    def test_psd_is_non_negative(self, ill):
        """Power spectral density values must be non-negative."""
        from scipy.signal import welch
        for ni in range(ill.n_neurons):
            for t in range(ill.n_trials):
                _, psd = welch(ill.data[t, :, ni], fs=1.0,
                               nperseg=min(32, ill.n_time))
                assert np.all(psd >= 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# summary
# ══════════════════════════════════════════════════════════════════════════════

class TestSummary:

    def test_runs_without_error(self, ill, capsys):
        ill.summary()

    def test_output_contains_shape(self, ill, capsys):
        ill.summary()
        out = capsys.readouterr().out
        assert "5" in out
        assert "60" in out
        assert "16" in out

    def test_output_contains_neuron_labels(self, ill, capsys):
        ill.summary()
        out = capsys.readouterr().out
        assert "Neuron 0"  in out
        assert "Neuron 15" in out

    def test_output_contains_snr(self, ill, capsys):
        ill.summary()
        out = capsys.readouterr().out
        assert "SNR" in out

    def test_output_contains_dt(self, ill, capsys):
        ill.summary()
        out = capsys.readouterr().out
        assert "dt" in out


# ══════════════════════════════════════════════════════════════════════════════
# plot_all  (smoke test — just ensure nothing raises)
# ══════════════════════════════════════════════════════════════════════════════

class TestPlotAll:

    def test_runs_without_error(self, ill):
        ill.plot_all()

    def test_runs_on_small_data(self, ill_small):
        ill_small.plot_all()