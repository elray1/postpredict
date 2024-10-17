# Tests for postpredict.util.argsort_random_tiebreak

import numpy as np
from postpredict.util import argsort_random_tiebreak


def test_argsort_random_tiebreak():
    arr = np.array([2, 1, 3, 1, 2])
    
    valid_argsorts = {
        (1, 3, 0, 4, 2),
        (3, 1, 0, 4, 2),
        (1, 3, 4, 0, 2),
        (3, 1, 4, 0, 2)
    }
    
    rng = np.random.default_rng(42)
    actual_argsorts = [tuple(argsort_random_tiebreak(arr, rng).tolist()) for i in range(1000)]
    
    assert valid_argsorts == set(actual_argsorts)


def test_argsort_random_tiebreak_reproducible():
    arr = np.array([2, 1, 3, 1, 2])

    rng = np.random.default_rng(42)
    actual_argsorts_1 = [tuple(argsort_random_tiebreak(arr, rng).tolist()) for i in range(1000)]
    rng = np.random.default_rng(42)
    actual_argsorts_2 = [tuple(argsort_random_tiebreak(arr, rng).tolist()) for i in range(1000)]
    
    assert actual_argsorts_1 == actual_argsorts_2
