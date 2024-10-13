# Tests for postpredict.dependence.TimeDependencePostprocessor.apply_shuffle
#
# The test case is based on the example given in Fig 2 of
# Clark, Martyn, et al. "The Schaake shuffle: A method for reconstructing
# space–time variability in forecasted precipitation and temperature fields."
# Journal of Hydrometeorology 5.1 (2004): 243-262.
#
# However, we swap the roles of geographical units and time, as we are modeling
# time dependence whereas that example is modeling spatial dependence.

import polars as pl
import pytest
from datetime import datetime, timedelta
from polars.testing import assert_frame_equal
from postpredict.dependence import TimeDependencePostprocessor


@pytest.fixture
def obs_data():
    return pl.DataFrame({
        "location": ["a"] * 20 + ["b"] * 20,
        "age_group": (["young"] * 10 + ["old"] * 10) * 2,
        "date": [datetime.strptime("2020-01-01", "%Y-%m-%d") + timedelta(i) for i in range(10)] * 4,
        "value": list(range(10, 50))
    })


def test_build_train_X_y_positive_horizons(obs_data, monkeypatch):
    # we use monkeypatch to remove abstract methods from the
    # TimeDependencePostprocessor class, allowing us to create an object of
    # that class so as to test the non-abstract apply_shuffle method it defines.
    # See https://stackoverflow.com/a/77748100
    monkeypatch.setattr(TimeDependencePostprocessor, "__abstractmethods__", set())
    tdp = TimeDependencePostprocessor()
    tdp.df = obs_data
    tdp.key_cols = ["location", "age_group"]
    tdp.time_col = "date",
    tdp.obs_col = "value"
    tdp.feat_cols = ["location", "age_group", "date"]

    tdp._build_train_X_Y(1, 4)
    
    expected_train_X = pl.DataFrame({
        "location": ["a"] * 12 + ["b"] * 12,
        "age_group": (["young"] * 6 + ["old"] * 6) * 2,
        "date": [datetime.strptime("2020-01-01", "%Y-%m-%d") + timedelta(i) for i in range(6)] * 4
    })
    
    expected_train_Y = pl.DataFrame({
        "value_shift_p1": list(range(11, 17)) + list(range(21, 27)) + list(range(31, 37)) + list(range(41, 47)),
        "value_shift_p2": list(range(12, 18)) + list(range(22, 28)) + list(range(32, 38)) + list(range(42, 48)),
        "value_shift_p3": list(range(13, 19)) + list(range(23, 29)) + list(range(33, 39)) + list(range(43, 49)),
        "value_shift_p4": list(range(14, 20)) + list(range(24, 30)) + list(range(34, 40)) + list(range(44, 50)),
    })
    
    assert_frame_equal(tdp.train_X, expected_train_X)
    assert_frame_equal(tdp.train_Y, expected_train_Y)


def test_build_train_X_y_nonnegative_horizons(obs_data, monkeypatch):
    # we use monkeypatch to remove abstract methods from the
    # TimeDependencePostprocessor class, allowing us to create an object of
    # that class so as to test the non-abstract apply_shuffle method it defines.
    # See https://stackoverflow.com/a/77748100
    monkeypatch.setattr(TimeDependencePostprocessor, "__abstractmethods__", set())
    tdp = TimeDependencePostprocessor()
    tdp.df = obs_data
    tdp.key_cols = ["location", "age_group"]
    tdp.time_col = "date",
    tdp.obs_col = "value"
    tdp.feat_cols = ["location", "age_group", "date"]

    tdp._build_train_X_Y(0, 4)
    
    expected_train_X = pl.DataFrame({
        "location": ["a"] * 12 + ["b"] * 12,
        "age_group": (["young"] * 6 + ["old"] * 6) * 2,
        "date": [datetime.strptime("2020-01-01", "%Y-%m-%d") + timedelta(i) for i in range(6)] * 4
    })
    
    expected_train_Y = pl.DataFrame({
        "value_shift_p0": list(range(10, 16)) + list(range(20, 26)) + list(range(30, 36)) + list(range(40, 46)),
        "value_shift_p1": list(range(11, 17)) + list(range(21, 27)) + list(range(31, 37)) + list(range(41, 47)),
        "value_shift_p2": list(range(12, 18)) + list(range(22, 28)) + list(range(32, 38)) + list(range(42, 48)),
        "value_shift_p3": list(range(13, 19)) + list(range(23, 29)) + list(range(33, 39)) + list(range(43, 49)),
        "value_shift_p4": list(range(14, 20)) + list(range(24, 30)) + list(range(34, 40)) + list(range(44, 50)),
    })
    
    assert_frame_equal(tdp.train_X, expected_train_X)
    assert_frame_equal(tdp.train_Y, expected_train_Y)


def test_build_train_X_y_negative_horizons(obs_data, monkeypatch):
    # we use monkeypatch to remove abstract methods from the
    # TimeDependencePostprocessor class, allowing us to create an object of
    # that class so as to test the non-abstract apply_shuffle method it defines.
    # See https://stackoverflow.com/a/77748100
    monkeypatch.setattr(TimeDependencePostprocessor, "__abstractmethods__", set())
    tdp = TimeDependencePostprocessor()
    tdp.df = obs_data
    tdp.key_cols = ["location", "age_group"]
    tdp.time_col = "date",
    tdp.obs_col = "value"
    tdp.feat_cols = ["location", "date"]

    tdp._build_train_X_Y(-1, 4)
    
    expected_train_X = pl.DataFrame({
        "location": ["a"] * 10 + ["b"] * 10,
        "date": [datetime.strptime("2020-01-01", "%Y-%m-%d") + timedelta(i) for i in range(1, 6)] * 4
    })
    
    expected_train_Y = pl.DataFrame({
        "value_shift_m1": list(range(10, 15)) + list(range(20, 25)) + list(range(30, 35)) + list(range(40, 45)),
        "value_shift_p0": list(range(11, 16)) + list(range(21, 26)) + list(range(31, 36)) + list(range(41, 46)),
        "value_shift_p1": list(range(12, 17)) + list(range(22, 27)) + list(range(32, 37)) + list(range(42, 47)),
        "value_shift_p2": list(range(13, 18)) + list(range(23, 28)) + list(range(33, 38)) + list(range(43, 48)),
        "value_shift_p3": list(range(14, 19)) + list(range(24, 29)) + list(range(34, 39)) + list(range(44, 49)),
        "value_shift_p4": list(range(15, 20)) + list(range(25, 30)) + list(range(35, 40)) + list(range(45, 50)),
    })
    
    assert_frame_equal(tdp.train_X, expected_train_X)
    assert_frame_equal(tdp.train_Y, expected_train_Y)
