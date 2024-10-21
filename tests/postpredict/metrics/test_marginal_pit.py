# Tests for postpredict.metrics.marginal_pit

from datetime import datetime

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal
from postpredict.metrics import marginal_pit


def test_marginal_pit():
    rng = np.random.default_rng(seed=123)
    model_out_wide = pl.concat([
        pl.DataFrame({
            "location": "a",
            "date": datetime.strptime("2024-10-01", "%Y-%m-%d"),
            "output_type": "sample",
            "output_type_id": np.linspace(0, 99, 100),
            "horizon1": rng.permutation(np.linspace(0, 9, 100)),
            "horizon2": rng.permutation(np.linspace(8, 17, 100)),
            "horizon3": rng.permutation(np.linspace(5.1, 16.1, 100))
        }),
        pl.DataFrame({
            "location": "b",
            "date": datetime.strptime("2024-10-08", "%Y-%m-%d"),
            "output_type": "sample",
            "output_type_id": np.linspace(100, 199, 100),
            "horizon1": rng.permutation(np.linspace(10.0, 19.0, 100)),
            "horizon2": rng.permutation(np.linspace(-3.0, 6.0, 100)),
            "horizon3": rng.permutation(np.linspace(10.99, 19.99, 100))
        })
    ])
    obs_data_wide = pl.DataFrame({
        "location": ["a", "a", "b", "b"],
        "date": [datetime.strptime("2024-10-01", "%Y-%m-%d"),
                 datetime.strptime("2024-10-08", "%Y-%m-%d"),
                 datetime.strptime("2024-10-01", "%Y-%m-%d"),
                 datetime.strptime("2024-10-08", "%Y-%m-%d")],
        "value": [3.0, 4.0, 0.0, 7.2],
        "value_lead1": [4.0, 10.0, 7.2, 9.6],
        "value_lead2": [10.0, 5.0, 9.6, 10.0],
        "value_lead3": [5.0, 2.0, 10.0, 14.1]
    })
    
    # expected PIT values: the number of samples less than or equal to
    # corresponding observed values
    expected_scores_df = pl.DataFrame({
        "location": ["a", "b"],
        "date": [datetime.strptime("2024-10-01", "%Y-%m-%d"),
                 datetime.strptime("2024-10-08", "%Y-%m-%d")],
        "pit_horizon1": [0.45, 0.0],
        "pit_horizon2": [0.23, 1.0],
        "pit_horizon3": [0.0, 0.35]
    })
    
    actual_scores_df = marginal_pit(model_out_wide=model_out_wide,
                                    obs_data_wide=obs_data_wide,
                                    key_cols=["location", "date"],
                                    pred_cols=["horizon1", "horizon2", "horizon3"],
                                    obs_cols=["value_lead1", "value_lead2", "value_lead3"],
                                    reduce_mean=False)
    
    assert_frame_equal(actual_scores_df, expected_scores_df, check_row_order=False, atol=1e-19)
