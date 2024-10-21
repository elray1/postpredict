import numpy as np
import polars as pl
from sklearn.metrics import pairwise_distances


def energy_score(model_out_wide: pl.DataFrame, obs_data_wide: pl.DataFrame,
                 key_cols: list[str] | None, pred_cols: list[str], obs_cols: list[str],
                 reduce_mean: bool = True) -> float | pl.DataFrame:
    """
    Compute the energy score for a collection of predictive samples.
    
    Parameters
    ----------
    model_out_wide: pl.DataFrame
        DataFrame of model outputs where each row corresponds to one
        (multivariate) sample from a multivariate distribution for one
        observational unit.
    obs_data_wide: pl.DataFrame
        DataFrame of observed values where each row corresponds to one
        (multivariate) observed outcome for one observational unit.
    key_cols: list[str]
        Columns that appear in both `model_out_wide` and `obs_data_wide` that
        identify observational units.
    pred_cols: list[str]
        Columns that appear in `model_out_wide` and identify predicted (sampled)
        values. The order of these should match the order of `obs_cols`.
    obs_cols: list[str]
        Columns that appear in `obs_data_wide` and identify observed values. The
        order of these should match the order of `pred_cols`.
    reduce_mean: bool = True
        Indicator of whether to return a numeric mean energy score (default) or
        a pl.DataFrame with one row per observational unit.
    
    Returns
    -------
    Either the mean energy score across all observational units (default) or a
    pl.DataFrame with one row per observational unit and scores stored in a
    column named `energy_score`.
    
    Notes
    -----
    We perform the energy score calculation of Eq. (7), p. 223 in
    Gneiting, T., Stanberry, L.I., Grimit, E.P. et al. Assessing probabilistic
    forecasts of multivariate quantities, with an application to ensemble
    predictions of surface winds. TEST 17, 211–235 (2008).
    https://doi.org/10.1007/s11749-008-0114-x
    https://link.springer.com/article/10.1007/s11749-008-0114-x
    """
    def energy_score_one_unit(df: pl.DataFrame):
        """
        Compute energy score for one observational unit based on a collection of
        samples. Note, we define this function here so that key_cols, pred_cols
        and obs_cols are in scope.
        """
        if df[pred_cols + obs_cols].null_count().to_numpy().sum() > 0:
            # Return np.nan rather than None here to avoid a rare schema
            # error when the first processed group would yield None.
            score = np.nan
        else:
            score = np.mean(pairwise_distances(df[pred_cols], df[0, obs_cols])) \
                - 0.5 * np.mean(pairwise_distances(df[pred_cols]))
        
        return df[0, key_cols].with_columns(energy_score = pl.lit(score))
    
    scores_by_unit = (
        model_out_wide
        .join(obs_data_wide, on = key_cols)
        .group_by(*key_cols)
        .map_groups(energy_score_one_unit)
    )
    
    if not reduce_mean:
        return scores_by_unit
    
    # replace NaN with None to average only across non-missing values
    return scores_by_unit["energy_score"].fill_nan(None).mean()


def marginal_pit(model_out_wide: pl.DataFrame, obs_data_wide: pl.DataFrame,
                 key_cols: list[str] | None, pred_cols: list[str], obs_cols: list[str],
                 reduce_mean: bool = True) -> float | pl.DataFrame:
    """
    Compute the probability integral transform (PIT) value for each of a
    collection of marginal predictive distributions represented by a set of
    samples.
    
    Parameters
    ----------
    model_out_wide: pl.DataFrame
        DataFrame of model outputs where each row corresponds to one
        (multivariate) sample from a multivariate distribution for one
        observational unit.
    obs_data_wide: pl.DataFrame
        DataFrame of observed values where each row corresponds to one
        (multivariate) observed outcome for one observational unit.
    key_cols: list[str]
        Columns that appear in both `model_out_wide` and `obs_data_wide` that
        identify observational units.
    pred_cols: list[str]
        Columns that appear in `model_out_wide` and identify predicted (sampled)
        values. The order of these should match the order of `obs_cols`.
    obs_cols: list[str]
        Columns that appear in `obs_data_wide` and identify observed values. The
        order of these should match the order of `pred_cols`.
    reduce_mean: bool = True
        Indicator of whether to return a numeric mean energy score (default) or
        a pl.DataFrame with one row per observational unit.
    
    Returns
    -------
    A pl.DataFrame with one row per observational unit and PIT values stored in
    columns named according to `[f"pit_{c}" for c in pred_cols]`.
    
    Notes
    -----
    Here, the PIT value is calculated as the proportion of samples that are less
    than or equal to the observed value.
    """
    scores_by_unit = (
        model_out_wide
        .join(obs_data_wide, on = key_cols)
        .group_by(key_cols)
        .agg(
            [
                (pl.col(pred_c) <= pl.col(obs_c)).mean().alias(f"pit_{pred_c}") \
                for pred_c, obs_c in zip(pred_cols, obs_cols)
            ]
        )
        .select(key_cols + [f"pit_{pred_c}" for pred_c in pred_cols])
    )

    return scores_by_unit
