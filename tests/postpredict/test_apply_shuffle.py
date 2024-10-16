# Tests for postpredict.dependence.TimeDependencePostprocessor.apply_shuffle

import polars as pl
from polars.testing import assert_frame_equal
from postpredict.dependence import TimeDependencePostprocessor


def test_apply_shuffle(wide_model_out, templates, monkeypatch):
    # we use monkeypatch to remove abstract methods from the
    # TimeDependencePostprocessor class, allowing us to create an object of
    # that class so as to test the non-abstract apply_shuffle method it defines.
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
    expected_final = pl.DataFrame({
        "location": ["a"] * 10,
        "population": [100.0] * 10,
        "age_group": ["young"] * 10,
        "output_type": ["sample"] * 10,
        "output_type_id": list(range(10)),
        "horizon1": [10.1, 8.8, 7.5, 10.3, 11.9, 15.3, 8.3, 9.7, 11.2, 12.5],
        "horizon2": [9.3, 7.2, 6.3, 8.6, 13.5, 17.7, 7.9, 7.5, 11.8, 12.2],
        "horizon3": [14.5, 15.6, 12.4, 16.3, 18.3, 23.9, 14.2, 13.5, 15.9, 17.6]
    })
    assert_frame_equal(actual_final, expected_final)
