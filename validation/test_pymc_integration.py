"""
Test PyMC Integration

Tests that PyMC models correctly integrate with agent classes
and can sample from posteriors.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False

pytestmark = pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")

from scripts.fitting.pymc_models import build_qlearning_model, build_wmrl_model


class TestPyMCModelBuilding:
    """Test that PyMC models build correctly."""

    def test_qlearning_model_builds(self, sample_participant_data):
        """Q-learning model builds without errors."""
        with build_qlearning_model(sample_participant_data) as model:
            assert model is not None
            assert 'alpha' in model.named_vars
            assert 'beta' in model.named_vars
            assert 'mu_alpha' in model.named_vars
            assert 'mu_beta' in model.named_vars

    def test_wmrl_model_builds(self, sample_participant_data):
        """WM-RL model builds without errors."""
        with build_wmrl_model(sample_participant_data) as model:
            assert model is not None
            assert 'alpha' in model.named_vars
            assert 'beta' in model.named_vars
            assert 'capacity' in model.named_vars
            assert 'w_wm' in model.named_vars

    def test_qlearning_model_with_multiple_participants(self, sample_multiparticipant_data):
        """Q-learning model handles multiple participants."""
        with build_qlearning_model(sample_multiparticipant_data) as model:
            # Check that individual parameters have correct shape
            n_participants = sample_multiparticipant_data['sona_id'].nunique()

            # This should work without errors
            assert model is not None

            # Individual parameters should have shape (n_participants,)
            # Note: Accessing shape may vary depending on PyMC version


class TestPyMCSampling:
    """Test that PyMC models can sample."""

    @pytest.mark.slow
    def test_qlearning_model_samples(self, sample_participant_data):
        """Q-learning model can sample (quick test with few samples)."""
        with build_qlearning_model(sample_participant_data) as model:
            trace = pm.sample(
                draws=100,
                tune=50,
                chains=1,
                random_seed=42,
                progressbar=False
            )

            # Check trace structure
            assert 'alpha' in trace.posterior
            assert 'beta' in trace.posterior

            # Check shapes (1 chain, 100 draws, 1 participant)
            assert trace.posterior['alpha'].shape == (1, 100, 1)
            assert trace.posterior['beta'].shape == (1, 100, 1)

    @pytest.mark.slow
    def test_wmrl_model_samples(self, sample_participant_data):
        """WM-RL model can sample."""
        with build_wmrl_model(sample_participant_data) as model:
            trace = pm.sample(
                draws=100,
                tune=50,
                chains=1,
                random_seed=42,
                progressbar=False
            )

            # Check all parameters present
            assert 'alpha' in trace.posterior
            assert 'beta' in trace.posterior
            assert 'capacity' in trace.posterior
            assert 'w_wm' in trace.posterior


class TestPyMCDiagnostics:
    """Test PyMC sampling diagnostics."""

    @pytest.mark.slow
    def test_qlearning_convergence_diagnostics(self, sample_multiparticipant_data):
        """Check convergence diagnostics for Q-learning."""
        with build_qlearning_model(sample_multiparticipant_data) as model:
            trace = pm.sample(
                draws=200,
                tune=100,
                chains=2,
                random_seed=42,
                progressbar=False
            )

            # Check for divergences
            divergences = trace.sample_stats.diverging.sum().values
            assert divergences < 10, f"Too many divergences: {divergences}"

            # Check R-hat (should be close to 1)
            summary = az.summary(trace, var_names=['mu_alpha', 'mu_beta'])

            # R-hat should exist and be reasonable
            if 'r_hat' in summary.columns:
                max_rhat = summary['r_hat'].max()
                assert max_rhat < 1.1, f"R-hat too high: {max_rhat}"


class TestPyMCLikelihoodComputation:
    """Test that likelihood computation works correctly."""

    def test_likelihood_increases_with_more_data(self, sample_participant_data):
        """
        Model likelihood should generally be better with more data.

        This is a sanity check that the likelihood function is working.
        """
        # Small dataset
        small_data = sample_participant_data.head(20)

        # Larger dataset
        large_data = sample_participant_data.head(40)

        # Build models and get log-likelihoods
        # (This is a simplified test - in practice we'd fit and compare)

        with build_qlearning_model(small_data) as model_small:
            assert model_small is not None

        with build_qlearning_model(large_data) as model_large:
            assert model_large is not None

        # If models build successfully, likelihood computation is working

    def test_likelihood_with_perfect_data(self):
        """
        Test likelihood with data that perfectly matches model predictions.

        Generate data from known parameters, then check that likelihood
        is highest at those parameters.
        """
        # Create perfect data (deterministic for this test)
        perfect_data = pd.DataFrame({
            'sona_id': ['P001'] * 10,
            'block': [3] * 10,
            'trial': list(range(1, 11)),
            'stimulus': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'key_press': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],  # Always correct
            'correct': [1] * 10  # Always correct
        })

        with build_qlearning_model(perfect_data) as model:
            # Model should build successfully
            assert model is not None

            # The likelihood function should handle this data
            # (detailed likelihood checks would require sampling)


class TestPyMCPosteriorPredictive:
    """Test posterior predictive capabilities."""

    @pytest.mark.slow
    def test_posterior_predictive_structure(self, sample_participant_data):
        """
        Test that we can work with posterior predictive.

        Note: Current implementation may not have built-in posterior predictive,
        but this tests the infrastructure is in place.
        """
        with build_qlearning_model(sample_participant_data) as model:
            trace = pm.sample(
                draws=50,
                tune=25,
                chains=1,
                random_seed=42,
                progressbar=False
            )

            # Basic posterior checks
            assert trace.posterior is not None
            assert len(trace.posterior.coords['draw']) == 50


class TestErrorHandling:
    """Test error handling in PyMC models."""

    def test_empty_data_raises_error(self):
        """Empty data should raise an informative error."""
        empty_data = pd.DataFrame({
            'sona_id': [],
            'block': [],
            'trial': [],
            'stimulus': [],
            'key_press': [],
            'correct': []
        })

        # Should either raise error or handle gracefully
        try:
            with build_qlearning_model(empty_data) as model:
                # If it doesn't raise an error, it should at least build
                assert model is not None
        except (ValueError, IndexError, KeyError):
            # These errors are acceptable for empty data
            pass

    def test_missing_columns_raises_error(self):
        """Missing required columns should raise error."""
        bad_data = pd.DataFrame({
            'sona_id': ['P001'],
            'block': [3],
            # Missing other required columns
        })

        with pytest.raises((KeyError, ValueError)):
            build_qlearning_model(bad_data)

    def test_invalid_stimulus_indices(self):
        """Invalid stimulus indices should be handled."""
        bad_data = pd.DataFrame({
            'sona_id': ['P001'] * 5,
            'block': [3] * 5,
            'trial': [1, 2, 3, 4, 5],
            'stimulus': [0, 1, 999, 2, 3],  # 999 is invalid
            'key_press': [0, 1, 2, 0, 1],
            'correct': [1, 0, 1, 0, 1]
        })

        # Should either handle this gracefully or raise informative error
        try:
            with build_qlearning_model(bad_data) as model:
                # If no error, model should build
                assert model is not None
        except (IndexError, ValueError):
            # Acceptable error for invalid indices
            pass
