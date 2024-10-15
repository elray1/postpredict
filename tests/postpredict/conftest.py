from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def wide_model_out():
    return pl.DataFrame({
        "location": ["a"] * 10,
        "age_group": ["young"] * 10,
        "output_type": ["sample"] * 10,
        "output_type_id": list(range(10)),
        "horizon1": [15.3, 11.2, 8.8, 11.9, 7.5, 9.7, 8.3, 12.5, 10.3, 10.1],
        "horizon2": [9.3, 6.3, 7.9, 7.5, 13.5, 11.8, 8.6, 17.7, 7.2, 12.2],
        "horizon3": [17.6, 15.6, 13.5, 14.2, 18.3, 15.9, 14.5, 23.9, 12.4, 16.3]
    })

@pytest.fixture
def long_model_out(wide_model_out):
    return (
        wide_model_out
        .unpivot(
            ["horizon1", "horizon2", "horizon3"],
            index=["location", "age_group", "output_type", "output_type_id"],
            variable_name="horizon"
        )
        .with_columns(horizon=pl.col("horizon").str.slice(-1, 1).cast(int))
    )

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
def obs_data():
    return pl.DataFrame({
        "location": ["a"] * 20 + ["b"] * 20,
        "age_group": (["young"] * 10 + ["old"] * 10) * 2,
        "date": [datetime.strptime("2020-01-01", "%Y-%m-%d") + timedelta(i) for i in range(10)] * 4,
        "value": list(range(10, 50))
    })
