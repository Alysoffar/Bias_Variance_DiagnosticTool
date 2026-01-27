import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from diagnoser.diagnosis.rules import diagnosis_rules
from diagnoser.diagnosis.recommendations import recommendation


class TestDiagnose:
    """Test suite for bias/variance diagnosis rules."""

    def test_high_bias_scenario(self):
        """Test detection of high bias (underfitting)."""
        # High train error, high val error, small gap
        train_errors = [0.35, 0.34, 0.33, 0.32]
        val_errors = [0.37, 0.36, 0.35, 0.34]
        config = {
            "diagnosis": {
                "gap_threshold": 0.1,
                "high_error_threshold": 0.3
            }
        }
        
        diagnosis = diagnosis_rules(train_errors, val_errors, config)
        assert diagnosis["label"] == "high_bias", "Should detect high bias"
        assert "train_error" in diagnosis
        assert "val_error" in diagnosis
        assert "gap" in diagnosis

    def test_high_variance_scenario(self):
        """Test detection of high variance (overfitting)."""
        # Low train error, high val error, large gap
        train_errors = [0.05, 0.03, 0.02, 0.01]
        val_errors = [0.25, 0.28, 0.30, 0.32]
        config = {
            "diagnosis": {
                "gap_threshold": 0.1,
                "high_error_threshold": 0.3
            }
        }
        
        diagnosis = diagnosis_rules(train_errors, val_errors, config)
        assert diagnosis["label"] == "high_variance", "Should detect high variance"
        assert diagnosis["gap"] > config["diagnosis"]["gap_threshold"]

    def test_balanced_scenario(self):
        """Test detection of balanced model."""
        # Low train error, low val error, small gap
        train_errors = [0.10, 0.08, 0.07, 0.06]
        val_errors = [0.12, 0.11, 0.10, 0.09]
        config = {
            "diagnosis": {
                "gap_threshold": 0.1,
                "high_error_threshold": 0.3
            }
        }
        
        diagnosis = diagnosis_rules(train_errors, val_errors, config)
        assert diagnosis["label"] == "balanced", "Should detect balanced model"

    def test_edge_case_empty_errors(self):
        """Test handling of empty error lists."""
        train_errors = []
        val_errors = []
        config = {"diagnosis": {"gap_threshold": 0.1, "high_error_threshold": 0.3}}
        
        diagnosis = diagnosis_rules(train_errors, val_errors, config)
        # Should handle gracefully, maybe return unknown or raise
        assert "label" in diagnosis

    def test_edge_case_single_point(self):
        """Test with single data point."""
        train_errors = [0.15]
        val_errors = [0.20]
        config = {"diagnosis": {"gap_threshold": 0.1, "high_error_threshold": 0.3}}
        
        diagnosis = diagnosis_rules(train_errors, val_errors, config)
        assert "label" in diagnosis
        assert "gap" in diagnosis

    def test_threshold_boundary(self):
        """Test behavior at threshold boundaries."""
        # Exactly at gap threshold
        train_errors = [0.10]
        val_errors = [0.20]  # gap = 0.10
        config = {"diagnosis": {"gap_threshold": 0.1, "high_error_threshold": 0.3}}
        
        diagnosis = diagnosis_rules(train_errors, val_errors, config)
        assert "label" in diagnosis

    def test_decreasing_errors(self):
        """Test with decreasing error trend."""
        train_errors = [0.5, 0.3, 0.2, 0.15]
        val_errors = [0.55, 0.35, 0.25, 0.20]
        config = {"diagnosis": {"gap_threshold": 0.1, "high_error_threshold": 0.3}}
        
        diagnosis = diagnosis_rules(train_errors, val_errors, config)
        assert diagnosis["label"] in ["high_bias", "high_variance", "balanced"]


class TestRecommendations:
    """Test suite for recommendation generation."""

    def test_high_bias_recommendation(self):
        """Test recommendation for high bias."""
        rec = recommendation("high_bias")
        assert isinstance(rec, str)
        assert len(rec) > 0
        # Should suggest increasing capacity
        assert any(keyword in rec.lower() for keyword in 
                   ["capacity", "features", "complex", "regularization"])

    def test_high_variance_recommendation(self):
        """Test recommendation for high variance."""
        rec = recommendation("high_variance")
        assert isinstance(rec, str)
        assert len(rec) > 0
        # Should suggest regularization or more data
        assert any(keyword in rec.lower() for keyword in 
                   ["data", "regularization", "simpl"])

    def test_balanced_recommendation(self):
        """Test recommendation for balanced model."""
        rec = recommendation("balanced")
        assert isinstance(rec, str)
        assert len(rec) > 0

    def test_unknown_label(self):
        """Test handling of unknown diagnosis label."""
        rec = recommendation("unknown_label")
        assert isinstance(rec, str)
        # Should provide some default or indicate unknown

    def test_all_labels_covered(self):
        """Test that all expected labels have recommendations."""
        labels = ["high_bias", "high_variance", "balanced"]
        for label in labels:
            rec = recommendation(label)
            assert isinstance(rec, str) and len(rec) > 0, f"Missing recommendation for {label}"
