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
from postpredict.dependence import TimeDependencePostprocessor


def test_pivot_horizon_positive_horizon(long_model_out, monkeypatch):
    # we use monkeypatch to remove abstract methods from the
    # TimeDependencePostprocessor class, allowing us to create an object of
    # that class so as to test the non-abstract apply_shuffle method it defines.
    # See https://stackoverflow.com/a/77748100
    monkeypatch.setattr(TimeDependencePostprocessor, "__abstractmethods__", set())
    tdp = TimeDependencePostprocessor()
    
    model_out = pl.concat([
        long_model_out.with_columns(
            population = 100
        ),
        long_model_out.with_columns(
            age_group = pl.lit("old"),
            value = pl.col("value") + 4,
            population = 150
        ),
        long_model_out.with_columns(
            location = pl.lit("b"),
            value = pl.col("value") + 12,
            population = 200
        ),
        long_model_out.with_columns(
            location = pl.lit("b"),
            age_group = pl.lit("old"),
            value = pl.col("value") - 2,
            population = 250
        )
    ])
    
    tdp.key_cols = ["location", "age_group"]
    tdp.time_col = "date",
    tdp.obs_col = "value"
    tdp.feat_cols = ["location", "age_group", "population"]
    wide_model_out = tdp._pivot_horizon(
        model_out=model_out,
        horizon_col="horizon",
        idx_col="output_type_id",
        pred_col="value"
    )
    
    # expected columns: everything in tdp.key_cols + tdp.feat_cols + "output_type_id" + f"postpredict_horizon{h}"
    expected_cols = set(tdp.key_cols + tdp.feat_cols + ["output_type_id"] + [f"postpredict_horizon{h}" for h in range(1, 4)])
    assert set(wide_model_out.columns) == expected_cols
    
    # same values within each horizon/group
    for location in ["a", "b"]:
        for age_group in ["young", "old"]:
            expected_values = (
                model_out
                .filter((pl.col("location") == location) & (pl.col("age_group") == age_group))
                [:, "value"]
                .to_numpy()
                .flatten()
                .tolist()
            )
            actual_values = (
                wide_model_out
                .filter((pl.col("location") == location) & (pl.col("age_group") == age_group))
                [:, [f"postpredict_horizon{h}" for h in range(1, 4)]]
                .to_numpy()
                .flatten()
                .tolist()
            )
            
            assert set(actual_values) == set(expected_values)
    
    # output_type_id different across different rows
    assert all(wide_model_out["output_type_id"].value_counts()["count"] == 1)



def test_pivot_horizon_negative_horizon(long_model_out, monkeypatch):
    # we use monkeypatch to remove abstract methods from the
    # TimeDependencePostprocessor class, allowing us to create an object of
    # that class so as to test the non-abstract apply_shuffle method it defines.
    # See https://stackoverflow.com/a/77748100
    monkeypatch.setattr(TimeDependencePostprocessor, "__abstractmethods__", set())
    tdp = TimeDependencePostprocessor()
    
    model_out = pl.concat([
        long_model_out.with_columns(
            horizon = pl.col("horizon") - 2,
            population = 100
        ),
        long_model_out.with_columns(
            age_group = pl.lit("old"),
            value = pl.col("value") + 4,
            horizon = pl.col("horizon") - 2,
            population = 150
        ),
        long_model_out.with_columns(
            location = pl.lit("b"),
            value = pl.col("value") + 12,
            horizon = pl.col("horizon") - 2,
            population = 200
        ),
        long_model_out.with_columns(
            location = pl.lit("b"),
            age_group = pl.lit("old"),
            value = pl.col("value") - 2,
            horizon = pl.col("horizon") - 2,
            population = 250
        )
    ])
    
    tdp.key_cols = ["location", "age_group"]
    tdp.time_col = "date",
    tdp.obs_col = "value"
    tdp.feat_cols = ["location", "age_group", "population"]
    wide_model_out = tdp._pivot_horizon(
        model_out=model_out,
        horizon_col="horizon",
        idx_col="output_type_id",
        pred_col="value"
    )
    
    # expected columns: everything in tdp.key_cols + tdp.feat_cols + "output_type_id" + f"postpredict_horizon{h}"
    expected_cols = set(tdp.key_cols + tdp.feat_cols + ["output_type_id"] + [f"postpredict_horizon{h}" for h in range(-1, 2)])
    assert set(wide_model_out.columns) == expected_cols
    
    # same values within each horizon/group
    for location in ["a", "b"]:
        for age_group in ["young", "old"]:
            expected_values = (
                model_out
                .filter((pl.col("location") == location) & (pl.col("age_group") == age_group))
                [:, "value"]
                .to_numpy()
                .flatten()
                .tolist()
            )
            actual_values = (
                wide_model_out
                .filter((pl.col("location") == location) & (pl.col("age_group") == age_group))
                [:, [f"postpredict_horizon{h}" for h in range(-1, 2)]]
                .to_numpy()
                .flatten()
                .tolist()
            )
            
            assert set(actual_values) == set(expected_values)
    
    # output_type_id different across different rows
    assert all(wide_model_out["output_type_id"].value_counts()["count"] == 1)


def test_pivot_horizon_diff_sample_count_by_group(long_model_out, monkeypatch):
    # we use monkeypatch to remove abstract methods from the
    # TimeDependencePostprocessor class, allowing us to create an object of
    # that class so as to test the non-abstract apply_shuffle method it defines.
    # See https://stackoverflow.com/a/77748100
    monkeypatch.setattr(TimeDependencePostprocessor, "__abstractmethods__", set())
    tdp = TimeDependencePostprocessor()
    
    model_out = pl.concat([
        long_model_out.with_columns(
            population = 100
        ).filter(pl.col("output_type_id") < 5),
        long_model_out.with_columns(
            age_group = pl.lit("old"),
            value = pl.col("value") + 4,
            population = 150
        ).filter(pl.col("output_type_id") > 2),
        long_model_out.with_columns(
            location = pl.lit("b"),
            value = pl.col("value") + 12,
            population = 200
        ).filter(pl.col("output_type_id") < 7),
        long_model_out.with_columns(
            location = pl.lit("b"),
            age_group = pl.lit("old"),
            value = pl.col("value") - 2,
            population = 250
        ).filter(pl.col("output_type_id") < 4)
    ])
    
    tdp.key_cols = ["location", "age_group"]
    tdp.time_col = "date",
    tdp.obs_col = "value"
    tdp.feat_cols = ["location", "age_group", "population"]
    wide_model_out = tdp._pivot_horizon(
        model_out=model_out,
        horizon_col="horizon",
        idx_col="output_type_id",
        pred_col="value"
    )
    
    # expected columns: everything in tdp.key_cols + tdp.feat_cols + "output_type_id" + f"postpredict_horizon{h}"
    expected_cols = set(tdp.key_cols + tdp.feat_cols + ["output_type_id"] + [f"postpredict_horizon{h}" for h in range(1, 4)])
    assert set(wide_model_out.columns) == expected_cols
    
    # same values within each horizon/group
    for location in ["a", "b"]:
        for age_group in ["young", "old"]:
            expected_values = (
                model_out
                .filter((pl.col("location") == location) & (pl.col("age_group") == age_group))
                [:, "value"]
                .to_numpy()
                .flatten()
                .tolist()
            )
            actual_values = (
                wide_model_out
                .filter((pl.col("location") == location) & (pl.col("age_group") == age_group))
                [:, [f"postpredict_horizon{h}" for h in range(1, 4)]]
                .to_numpy()
                .flatten()
                .tolist()
            )
            
            assert set(actual_values) == set(expected_values)
    
    # output_type_id different across different rows
    assert all(wide_model_out["output_type_id"].value_counts()["count"] == 1)


def test_pivot_horizon_diff_sample_count_by_horizon_same_group_errors(long_model_out, monkeypatch):
    # we use monkeypatch to remove abstract methods from the
    # TimeDependencePostprocessor class, allowing us to create an object of
    # that class so as to test the non-abstract apply_shuffle method it defines.
    # See https://stackoverflow.com/a/77748100
    monkeypatch.setattr(TimeDependencePostprocessor, "__abstractmethods__", set())
    tdp = TimeDependencePostprocessor()
    
    model_out = pl.concat([
        long_model_out.with_columns(
            population = 100
        ).filter((pl.col("output_type_id") < 5) | ((pl.col("output_type_id") < 7) & (pl.col("horizon") == 2))),
        long_model_out.with_columns(
            age_group = pl.lit("old"),
            value = pl.col("value") + 4,
            population = 150
        ).filter(pl.col("output_type_id") > 2),
        long_model_out.with_columns(
            location = pl.lit("b"),
            value = pl.col("value") + 12,
            population = 200
        ).filter(pl.col("output_type_id") < 7),
        long_model_out.with_columns(
            location = pl.lit("b"),
            age_group = pl.lit("old"),
            value = pl.col("value") - 2,
            population = 250
        ).filter(pl.col("output_type_id") < 4)
    ])
    
    tdp.key_cols = ["location", "age_group"]
    tdp.time_col = "date",
    tdp.obs_col = "value"
    tdp.feat_cols = ["location", "age_group", "population"]
    with pytest.raises(ValueError):
        tdp._pivot_horizon(
            model_out=model_out,
            horizon_col="horizon",
            idx_col="output_type_id",
            pred_col="value"
        )


def test_pivot_horizon_missing_horizon_errors(long_model_out, monkeypatch):
    # we use monkeypatch to remove abstract methods from the
    # TimeDependencePostprocessor class, allowing us to create an object of
    # that class so as to test the non-abstract apply_shuffle method it defines.
    # See https://stackoverflow.com/a/77748100
    monkeypatch.setattr(TimeDependencePostprocessor, "__abstractmethods__", set())
    tdp = TimeDependencePostprocessor()
    
    model_out = pl.concat([
        long_model_out.with_columns(
            population = 100
        ).filter(pl.col("horizon") != 2),
        long_model_out.with_columns(
            age_group = pl.lit("old"),
            value = pl.col("value") + 4,
            population = 150
        ),
        long_model_out.with_columns(
            location = pl.lit("b"),
            value = pl.col("value") + 12,
            population = 200
        ),
        long_model_out.with_columns(
            location = pl.lit("b"),
            age_group = pl.lit("old"),
            value = pl.col("value") - 2,
            population = 250
        )
    ])
    
    tdp.key_cols = ["location", "age_group"]
    tdp.time_col = "date",
    tdp.obs_col = "value"
    tdp.feat_cols = ["location", "age_group", "population"]
    with pytest.raises(ValueError):
        tdp._pivot_horizon(
            model_out=model_out,
            horizon_col="horizon",
            idx_col="output_type_id",
            pred_col="value"
        )
