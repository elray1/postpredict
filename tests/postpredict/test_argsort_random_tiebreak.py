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
    
    actual_argsorts = [tuple(argsort_random_tiebreak(arr).tolist()) for i in range(1000)]
    
    assert valid_argsorts == set(actual_argsorts)
