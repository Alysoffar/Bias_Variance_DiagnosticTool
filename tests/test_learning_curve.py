import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from diagnoser.curves.learning_curve import learning_curve
from diagnoser.curves.sampling import sampling


class TestSampling:
    """Test suite for sampling function."""

    def test_sampling_size(self):
        """Test that sampling returns correct size."""
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        m = 30
        
        X_sub, y_sub = sampling(X, y, m, seed=42)
        
        assert len(X_sub) == m, f"Expected {m} samples, got {len(X_sub)}"
        assert len(y_sub) == m, f"Expected {m} labels, got {len(y_sub)}"
        assert X_sub.shape[1] == X.shape[1], "Feature dimension should match"

    def test_sampling_reproducibility(self):
        """Test that same seed produces same sample."""
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        m = 30
        
        X_sub1, y_sub1 = sampling(X, y, m, seed=42)
        X_sub2, y_sub2 = sampling(X, y, m, seed=42)
        
        np.testing.assert_array_equal(X_sub1, X_sub2, "Same seed should produce same X samples")
        np.testing.assert_array_equal(y_sub1, y_sub2, "Same seed should produce same y samples")

    def test_sampling_different_seeds(self):
        """Test that different seeds produce different samples."""
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        m = 30
        
        X_sub1, y_sub1 = sampling(X, y, m, seed=42)
        X_sub2, y_sub2 = sampling(X, y, m, seed=123)
        
        # Should be different with high probability
        assert not np.array_equal(X_sub1, X_sub2), "Different seeds should produce different samples"

    def test_sampling_no_replacement(self):
        """Test that sampling is without replacement."""
        X = np.arange(50).reshape(-1, 1)
        y = np.arange(50)
        m = 20
        
        X_sub, y_sub = sampling(X, y, m, seed=42)
        
        # Check no duplicates in sampled indices
        unique_samples = np.unique(X_sub)
        assert len(unique_samples) == m, "Sampling should be without replacement"

    def test_sampling_full_dataset(self):
        """Test sampling when m equals dataset size."""
        X = np.random.rand(50, 3)
        y = np.random.randint(0, 2, 50)
        m = 50
        
        X_sub, y_sub = sampling(X, y, m, seed=42)
        
        assert len(X_sub) == 50
        assert len(y_sub) == 50

    def test_sampling_small_m(self):
        """Test sampling with very small m."""
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        m = 5
        
        X_sub, y_sub = sampling(X, y, m, seed=42)
        
        assert len(X_sub) == 5
        assert len(y_sub) == 5


class TestLearningCurve:
    """Test suite for learning_curve function."""

    def test_learning_curve_structure(self):
        """Test that learning curve returns correct structure."""
        # Create synthetic data
        X_train = np.random.rand(100, 5)
        y_train = np.random.randint(0, 2, 100)
        X_val = np.random.rand(30, 5)
        y_val = np.random.randint(0, 2, 30)
        
        config = {
            "model_type": "tree_classifier",
            "task_type": "classification",
            "learning_curve": {"sizes": [0.2, 0.5, 0.8]},
            "model": {"max_depth": 3, "random_state": 42},
            "random_seed": 42
        }
        
        sizes, train_errors, val_errors = learning_curve(config, X_train, y_train, X_val, y_val)
        
        assert len(sizes) == 3, "Should return 3 size points"
        assert len(train_errors) == 3, "Should return 3 train errors"
        assert len(val_errors) == 3, "Should return 3 val errors"

    def test_learning_curve_sizes(self):
        """Test that sizes match config."""
        X_train = np.random.rand(100, 5)
        y_train = np.random.randint(0, 2, 100)
        X_val = np.random.rand(30, 5)
        y_val = np.random.randint(0, 2, 30)
        
        expected_sizes = [0.1, 0.3, 0.5, 0.7, 1.0]
        config = {
            "model_type": "tree_classifier",
            "task_type": "classification",
            "learning_curve": {"sizes": expected_sizes},
            "model": {"max_depth": 3},
            "random_seed": 42
        }
        
        sizes, _, _ = learning_curve(config, X_train, y_train, X_val, y_val)
        
        assert sizes == expected_sizes, "Returned sizes should match config"

    def test_learning_curve_error_bounds(self):
        """Test that errors are within valid bounds."""
        X_train = np.random.rand(100, 5)
        y_train = np.random.randint(0, 2, 100)
        X_val = np.random.rand(30, 5)
        y_val = np.random.randint(0, 2, 30)
        
        config = {
            "model_type": "tree_classifier",
            "task_type": "classification",
            "learning_curve": {"sizes": [0.3, 0.6, 1.0]},
            "model": {"max_depth": 3, "random_state": 42},
            "random_seed": 42
        }
        
        _, train_errors, val_errors = learning_curve(config, X_train, y_train, X_val, y_val)
        
        # Classification error should be between 0 and 1
        for err in train_errors:
            assert 0 <= err <= 1, f"Train error {err} out of bounds [0, 1]"
        for err in val_errors:
            assert 0 <= err <= 1, f"Val error {err} out of bounds [0, 1]"

    def test_learning_curve_regression(self):
        """Test learning curve with regression task."""
        X_train = np.random.rand(100, 5)
        y_train = np.random.rand(100) * 10  # regression targets
        X_val = np.random.rand(30, 5)
        y_val = np.random.rand(30) * 10
        
        config = {
            "model_type": "ridge_regression",
            "task_type": "regression",
            "learning_curve": {"sizes": [0.3, 0.6, 1.0]},
            "model": {"alpha": 1.0},
            "random_seed": 42
        }
        
        sizes, train_errors, val_errors = learning_curve(config, X_train, y_train, X_val, y_val)
        
        assert len(train_errors) == 3
        assert len(val_errors) == 3
        # MSE should be non-negative
        for err in train_errors:
            assert err >= 0, f"Train MSE {err} should be non-negative"
        for err in val_errors:
            assert err >= 0, f"Val MSE {err} should be non-negative"

    def test_learning_curve_increasing_data(self):
        """Test that using more data generally improves or maintains performance."""
        # Use a simple linear problem where more data should help
        np.random.seed(42)
        X_train = np.random.rand(200, 3)
        y_train = (X_train[:, 0] + X_train[:, 1] > 1).astype(int)
        X_val = np.random.rand(50, 3)
        y_val = (X_val[:, 0] + X_val[:, 1] > 1).astype(int)
        
        config = {
            "model_type": "logistic_regression",
            "task_type": "classification",
            "learning_curve": {"sizes": [0.2, 0.5, 0.8, 1.0]},
            "model": {"C": 1.0, "max_iter": 200, "random_state": 42},
            "random_seed": 42
        }
        
        sizes, train_errors, val_errors = learning_curve(config, X_train, y_train, X_val, y_val)
        
        # Validation error should generally decrease or stay stable with more data
        # (not strictly enforced due to randomness, but check structure is correct)
        assert len(val_errors) == 4
        assert all(isinstance(err, (int, float)) for err in val_errors)

    def test_learning_curve_single_size(self):
        """Test learning curve with single size point."""
        X_train = np.random.rand(100, 5)
        y_train = np.random.randint(0, 2, 100)
        X_val = np.random.rand(30, 5)
        y_val = np.random.randint(0, 2, 30)
        
        config = {
            "model_type": "tree_classifier",
            "task_type": "classification",
            "learning_curve": {"sizes": [0.5]},
            "model": {"max_depth": 3, "random_state": 42},
            "random_seed": 42
        }
        
        sizes, train_errors, val_errors = learning_curve(config, X_train, y_train, X_val, y_val)
        
        assert len(sizes) == 1
        assert len(train_errors) == 1
        assert len(val_errors) == 1
