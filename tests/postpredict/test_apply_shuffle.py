# Tests for postpredict.dependence.TimeDependencePostprocessor.apply_shuffle
#
# The test case is based on the example given in Fig 2 of
# Clark, Martyn, et al. "The Schaake shuffle: A method for reconstructing
# space–time variability in forecasted precipitation and temperature fields."
# Journal of Hydrometeorology 5.1 (2004): 243-262.
#
# However, we swap the roles of geographical units and time, as we are modeling
# time dependence whereas that example is modeling spatial dependence.

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from postpredict.dependence import TimeDependencePostprocessor


@pytest.fixture
def wide_model_out():
    return pl.DataFrame({
        "location": ["a"] * 10,
        "output_type": ["sample"] * 10,
        "output_type_id": list(range(10)),
        "horizon1": [15.3, 11.2, 8.8, 11.9, 7.5, 9.7, 8.3, 12.5, 10.3, 10.1],
        "horizon2": [9.3, 6.3, 7.9, 7.5, 13.5, 11.8, 8.6, 17.7, 7.2, 12.2],
        "horizon3": [17.6, 15.6, 13.5, 14.2, 18.3, 15.9, 14.5, 23.9, 12.4, 16.3]
    })


@pytest.fixture
def templates():
    return np.array([
        [10.7, 10.9, 13.5],
        [9.3, 9.1, 13.7],
        [6.8, 7.2, 9.3],
        [11.3, 10.7, 15.6],
        [12.2, 13.1, 17.8],
        [13.6, 14.2, 19.3],
        [8.9, 9.4, 12.1],
        [9.9, 9.2, 11.8],
        [11.8, 11.9, 15.2],
        [12.9, 12.5, 16.9]
    ])


@pytest.fixture
def expected_final():
    return pl.DataFrame({
        "location": ["a"] * 10,
        "output_type": ["sample"] * 10,
        "output_type_id": list(range(10)),
        "horizon1": [10.1, 8.8, 7.5, 10.3, 11.9, 15.3, 8.3, 9.7, 11.2, 12.5],
        "horizon2": [9.3, 7.2, 6.3, 8.6, 13.5, 17.7, 7.9, 7.5, 11.8, 12.2],
        "horizon3": [14.5, 15.6, 12.4, 16.3, 18.3, 23.9, 14.2, 13.5, 15.9, 17.6]
    })
    

def test_apply_shuffle(wide_model_out, templates, expected_final, monkeypatch):
    # we use monkeypatch to remove abstract methods from the
    # TimeDependencePostprocessor class, allowing us to create an object of
    # that class so as to test the non-abstract apply_shuffle method it defines.
    # See https://stackoverflow.com/a/77748100
    monkeypatch.setattr(TimeDependencePostprocessor, "__abstractmethods__", set())
    tdp = TimeDependencePostprocessor()
    actual_final = tdp._apply_shuffle(
        wide_model_out,
        [f"horizon{h}" for h in range(1, 4)],
        templates
    )
    assert_frame_equal(actual_final, expected_final)
