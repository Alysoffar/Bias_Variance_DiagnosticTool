import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from diagnoser.metrics.metrics import classification_error, regression_error


class TestClassificationError:
    """Test suite for classification_error function."""

    def test_perfect_prediction(self):
        """Test when all predictions are correct."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 1])
        error = classification_error(y_true, y_pred)
        assert error == 0.0, "Perfect predictions should have zero error"

    def test_all_wrong_prediction(self):
        """Test when all predictions are incorrect."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1])
        error = classification_error(y_true, y_pred)
        assert error == 1.0, "All wrong predictions should have error of 1.0"

    def test_half_correct(self):
        """Test when half the predictions are correct."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        error = classification_error(y_true, y_pred)
        assert error == 0.5, "Half correct should have error of 0.5"

    def test_multiclass(self):
        """Test with multiclass labels."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 1, 1, 1])
        error = classification_error(y_true, y_pred)
        expected = 2.0 / 6.0  # 2 wrong out of 6
        assert abs(error - expected) < 1e-6, f"Expected {expected}, got {error}"

    def test_single_sample(self):
        """Test with single sample."""
        y_true = np.array([1])
        y_pred = np.array([1])
        error = classification_error(y_true, y_pred)
        assert error == 0.0

        y_pred = np.array([0])
        error = classification_error(y_true, y_pred)
        assert error == 1.0


class TestRegressionError:
    """Test suite for regression_error function."""

    def test_perfect_prediction(self):
        """Test when all predictions are exact."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])
        error = regression_error(y_true, y_pred)
        assert error == 0.0, "Perfect predictions should have zero MSE"

    def test_constant_offset(self):
        """Test with constant offset."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([2.0, 3.0, 4.0, 5.0])  # all off by 1
        error = regression_error(y_true, y_pred)
        assert error == 1.0, "Constant offset of 1 should give MSE of 1.0"

    def test_known_mse(self):
        """Test with known MSE value."""
        y_true = np.array([0.0, 1.0, 2.0])
        y_pred = np.array([0.5, 1.5, 2.5])
        # MSE = mean((0.5^2, 0.5^2, 0.5^2)) = 0.25
        error = regression_error(y_true, y_pred)
        assert abs(error - 0.25) < 1e-6, f"Expected 0.25, got {error}"

    def test_single_sample(self):
        """Test with single sample."""
        y_true = np.array([5.0])
        y_pred = np.array([8.0])
        error = regression_error(y_true, y_pred)
        assert error == 9.0, "(8-5)^2 = 9"

    def test_negative_values(self):
        """Test with negative values."""
        y_true = np.array([-1.0, -2.0, -3.0])
        y_pred = np.array([-1.0, -2.0, -3.0])
        error = regression_error(y_true, y_pred)
        assert error == 0.0

    def test_large_errors(self):
        """Test that large errors are squared correctly."""
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([10.0, -10.0])
        # MSE = mean((100, 100)) = 100
        error = regression_error(y_true, y_pred)
        assert error == 100.0
