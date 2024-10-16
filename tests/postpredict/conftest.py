from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

# The test data is based on the example given in Fig 2 of
# Clark, Martyn, et al. "The Schaake shuffle: A method for reconstructing
# space–time variability in forecasted precipitation and temperature fields."
# Journal of Hydrometeorology 5.1 (2004): 243-262.
#
# However, we swap the roles of geographical units and time, as we are modeling
# time dependence whereas that example is modeling spatial dependence.
#
# The data for location "a", age_group "young" replicate the data from Fig 2.
# Data for other locations/age groups are obtained via different offsets.
#
# location and age_group are task ids, while horizon1, ... horizon3 correspond
# to a "horizon" task id variable that has been pivoted wider, with three
# horizon values.
# output_type and output_type_id are standard hubverse columns.
# population is an extra column to be used as a feature in weighting.

@pytest.fixture
def wide_model_out():
    wmo_group1 = pl.DataFrame({
        "location": ["a"] * 10,
        "population": [100.0] * 10,
        "age_group": ["young"] * 10,
        "output_type": ["sample"] * 10,
        "output_type_id": list(range(10)),
        "horizon1": [15.3, 11.2, 8.8, 11.9, 7.5, 9.7, 8.3, 12.5, 10.3, 10.1],
        "horizon2": [9.3, 6.3, 7.9, 7.5, 13.5, 11.8, 8.6, 17.7, 7.2, 12.2],
        "horizon3": [17.6, 15.6, 13.5, 14.2, 18.3, 15.9, 14.5, 23.9, 12.4, 16.3]
    })
    return pl.concat([
        wmo_group1,
        wmo_group1.with_columns(
            age_group = pl.lit("old"),
            horizon1 = pl.col("horizon1") + 4.0,
            horizon2 = pl.col("horizon2") + 4.0,
            horizon3 = pl.col("horizon3") + 4.0,
            population = 150.0
        ),
        wmo_group1.with_columns(
            location = pl.lit("b"),
            horizon1 = pl.col("horizon1") + 12.0,
            horizon2 = pl.col("horizon2") + 12.0,
            horizon3 = pl.col("horizon3") + 12.0,
            population = 200.0
        ),
        wmo_group1.with_columns(
            location = pl.lit("b"),
            age_group = pl.lit("old"),
            horizon1 = pl.col("horizon1") - 2.0,
            horizon2 = pl.col("horizon2") - 2.0,
            horizon3 = pl.col("horizon3") - 2.0,
            population = 250.0
        )
    ])

@pytest.fixture
def long_model_out(wide_model_out):
    return (
        wide_model_out
        .unpivot(
            ["horizon1", "horizon2", "horizon3"],
            index=["location", "population", "age_group", "output_type", "output_type_id"],
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
