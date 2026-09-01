"""Tests for the resilience index and classification logic."""

import numpy as np
import pytest

from src.models.evaluate import resilience_class, resilience_index


def test_resilience_index_ratio():
    actual = np.array([100.0, 90.0, 110.0])
    predicted = np.array([100.0, 100.0, 100.0])
    result = resilience_index(actual, predicted)
    np.testing.assert_allclose(result, [1.0, 0.9, 1.1])


def test_resilience_index_guards_zero_denominator():
    result = resilience_index(np.array([1.0]), np.array([0.0]))
    assert np.isnan(result[0])


@pytest.mark.parametrize(
    "value,expected",
    [
        (0.95, "Highly Resilient"),
        (0.9, "Highly Resilient"),
        (0.8, "Moderately Resilient"),
        (0.7, "Moderately Resilient"),
        (0.5, "Vulnerable"),
    ],
)
def test_resilience_class_boundaries(value, expected):
    assert resilience_class(value) == expected