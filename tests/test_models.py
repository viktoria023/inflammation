"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
from unittest.mock import patch
import pytest

def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""
    from inflammation.models import daily_mean

    # NB: the comment 'yapf: disable' disables automatic formatting using
    # a tool called 'yapf' which we have used when creating this project
    test_array = np.array([[0, 0],
                           [0, 0],
                           [0, 0]])  # yapf: disable

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(np.array([0, 0]), daily_mean(test_array))


def test_daily_mean_integers():
    """Test that mean function works for an array of positive integers."""
    from inflammation.models import daily_mean

    test_array = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])  # yapf: disable

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(np.array([3, 4]), daily_mean(test_array))

@patch('inflammation.models.get_data_dir', return_value='/data_dir')
def test_load_csv(mock_get_data_dir):
    from inflammation.models import load_csv
    with patch('numpy.loadtxt') as mock_loadtxt:
        load_csv('test.csv')
        name, args, kwargs = mock_loadtxt.mock_calls[0]
        assert kwargs['fname'] == '/data_dir/test.csv'
        load_csv('/test.csv')
        name, args, kwargs = mock_loadtxt.mock_calls[1]
        assert kwargs['fname'] == '/test.csv'

# TODO(lesson-automatic) Implement tests for the other statistical functions

@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [0, 0, 0]),
        ([[4, 2, 5], [1, 6, 2], [4, 1, 9]], [4, 6, 9]),
        ([[4, -2, 5], [1, -6, 2], [-4, -1, 9]], [4, -1, 9]),
    ])
def test_daily_max(test, expected):
    """Test max function works for zeroes, positive integers, mix of positive/negative integers."""
    from inflammation.models import daily_max
    npt.assert_array_equal(np.array(expected), daily_max(np.array(test)))


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [0, 0, 0]),
        ([[4, 2, 5], [1, 6, 2], [4, 1, 9]], [1, 1, 2]),
        ([[4, -2, 5], [1, -6, 2], [-4, -1, 9]], [-4, -6, 2]),
    ])
def test_daily_min(test, expected):
    """Test min function works for zeroes, positive integers, mix of positive/negative integers."""
    from inflammation.models import daily_min
    npt.assert_array_equal(np.array(expected), daily_min(np.array(test)))

@pytest.mark.parametrize(
    "test, expected, raises",
    [
        (
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            None,
        ),
        (
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            None,
        ),
        (
            [[float('nan'), 1, 1], [1, 1, 1], [1, 1, 1]],
            [[0, 1, 1], [1, 1, 1], [1, 1, 1]],
            None,
        ),
        (
            [[1, 2, 3], [4, 5, float('nan')], [7, 8, 9]],
            [[0.33, 0.66, 1], [0.8, 1, 0], [0.77, 0.88, 1]],
            None,
        ),
        (
            [[-1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[0, 0.66, 1], [0.66, 0.83, 1], [0.77, 0.88, 1]],
            None,
        ),
        (
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[0.33, 0.66, 1], [0.66, 0.83, 1], [0.77, 0.88, 1]],
            None,
        )
        (
            [[-1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[0, 0.66, 1], [0.66, 0.83, 1], [0.77, 0.88, 1]],
            ValueError,
        ),
        (
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[0.33, 0.66, 1], [0.66, 0.83, 1], [0.77, 0.88, 1]],
            None,
        ),
        (
            'hello',
            None,
            TypeError,
        ),
        (
            3,
            None,
            TypeError,
        ),
        (
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[0.33, 0.66, 1], [0.66, 0.83, 1], [0.77, 0.88, 1]],
            None,
        )
    ])
def test_patient_normalise(test, expected, raises):
    """Test normalisation works for arrays of one and positive integers."""
    from inflammation.models import patient_normalise
    if isinstance(test, list):
        test = np.array(test)
    if raises:
        with pytest.raises(raises):
            npt.assert_almost_equal(np.array(expected), patient_normalise(test), decimal=2)
    else:
        npt.assert_almost_equal(np.array(expected), patient_normalise(test), decimal=2)

@patch('inflammation.models.get_data_dir', return_value='/data_dir')
def test_load_csv(mock_get_data_dir):
   from inflammation.models import load_csv
   with patch('numpy.loadtxt') as mock_loadtxt:
       load_csv('test.csv')
       name, args, kwargs = mock_loadtxt.mock_calls[0]
       assert kwargs['fname'] == '/data_dir/test.csv'
       load_csv('/test.csv')
       name, args, kwargs = mock_loadtxt.mock_calls[1]
<<<<<<< HEAD
       assert kwargs['fname'] == '/test.csv'
=======
       assert kwargs['fname'] == '/test.csv'
>>>>>>> aeeed91a52bed9ab6aee912690f1ceaa47006ee2
