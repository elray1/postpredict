# Tests for postpredict.dependence.TimeDependencePostprocessor._apply_shuffle

import polars as pl
from polars.testing import assert_frame_equal
from postpredict.dependence import TimeDependencePostprocessor


def test_apply_shuffle(wide_model_out, templates, wide_expected_final, monkeypatch):
    # we use monkeypatch to remove abstract methods from the
    # TimeDependencePostprocessor class, allowing us to create an object of
    # that class so as to test the non-abstract _apply_shuffle method it defines.
    # See https://stackoverflow.com/a/77748100
    monkeypatch.setattr(TimeDependencePostprocessor, "__abstractmethods__", set())
    tdp = TimeDependencePostprocessor()
    
    # To match Fig 2 of Clark et al., we just keep the portion of the data for
    # location "a", age_group "young"
    actual_final = tdp._apply_shuffle(
        wide_model_out.filter((pl.col("location") == "a") & (pl.col("age_group") == "young")),
        [f"horizon{h}" for h in range(1, 4)],
        templates
    )
    assert_frame_equal(
        actual_final,
        wide_expected_final.filter((pl.col("location") == "a") & (pl.col("age_group") == "young"))
    )
