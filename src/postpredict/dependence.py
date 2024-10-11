import abc

import numpy as np
import polars as pl

from postpredict import weighters
from postpredict.util import argsort_random_tiebreak


class TimeDependencePostprocessor(abc.ABC):
    def __init__(self):
        pass


    @abc.abstractmethod
    def fit(self, **kwargs):
        """
        Fit a model for temporal dependence across prediction horizons

        Returns
        -------
        None
        """


    @abc.abstractmethod
    def transform(self, model_out, **kwargs):
        """
        Apply an estimated time dependence model to sample predictions to induce
        dependence across time in the predictive samples.
        
        Parameters
        ----------
        model_out: polars dataframe with sample predictions that do not
        necessarily capture temporal dependence.
        
        Returns
        -------
        A copy of the model_out parameter, with sample indices updated so that
        they reflect the estimated temporal dependence structure.
        """
    
    
    def apply_shuffle(self,
                      wide_model_out: pl.DataFrame,
                      value_cols: list[str],
                      templates: np.ndarray) -> pl.DataFrame:
        """
        Given a collection of samples and an equally-sized collection of
        "dependence templates", shuffle the samples to match the rankings in the
        dependence templates. It is assumed that samples are exchangeable,
        i.e. this function should be called with samples for a single
        observational unit (e.g., one location/age group combination).
        
        Parameters
        ----------
        wide_model_out: polars dataframe with sample predictions that do not
        necessarily capture temporal dependence.
        value_cols: character vector of columns in `wide_model_out` that contain
        predicted values over time. These should be given in temporal order.
        templates: numpy array of shape (wide_model_out.shape[0], len(value_cols))
        containing dependence templates.
        
        Returns
        -------
        A copy of the `wide_model_out` argument, with values in the `value_cols`
        columns shuffled to match the rankings in the `templates`.
        
        Notes
        -----
        The argument `wide_model_out` should be in "semi-wide" form, where each
        row corresponds to one sample for one observational unit. Here, an
        observational unit is defined by a combination of keys such as location
        and/or age group. For each such unit and sample, the predictive samples
        should be in a set of columns given in temporal order; for example,
        these might be called `horizon1` through `horizon4`.
        """
        col_orderings = {
            c: argsort_random_tiebreak(templates[:, i]) \
            for i, c in enumerate(value_cols)
        }

        shuffled_wmo = wide_model_out.clone()
        for c in value_cols:
            shuffled_wmo = shuffled_wmo.with_columns(pl.col(c).sort().alias(c))
            shuffled_wmo[col_orderings[c], c] = shuffled_wmo[c]
        
        return shuffled_wmo



class Schaake(TimeDependencePostprocessor):
    def __init__(self, weighter_class=weighters.EqualWeighter, **kwargs) -> None:
        self.weighter = weighter_class(**kwargs)


    def fit(self, df, key_cols=None, time_col="date", obs_col="value"):
        """
        Fit a Schaake shuffle model for temporal dependence across prediction
        horizons. In practice this just involves saving the input arguments for
        later use; the Schaake shuffle does not require any parameter estimation.
        
        Parameters
        ----------
        df: polars dataframe with training set observations.
        key_cols: names of columns in `df` used to identify observational units,
        e.g. location or age group.
        time_col: name of column in `df` that contains the time index.
        obs_col: name of column in `df` that contains observed values.
        
        Returns
        -------
        None
        """
        self.df = df
        self.key_cols = key_cols
        self.time_col = time_col
        self.obs_col = obs_col
        self.shift_varnames = []

    
    def transform(self, model_out, horizon_col="horizon"):
        """
        Apply the Schaake shuffle to sample predictions to induce dependence
        across time in the predictive samples.
        
        Parameters
        ----------
        model_out: polars dataframe with sample predictions that do not
        necessarily capture temporal dependence.
        horizon_col: name of column in model_out that records the prediction horizon
        
        Returns
        -------
        A copy of the model_out parameter, with sample indices updated so that
        they reflect the estimated temporal dependence structure.
        """
        self.horizon_col = horizon_col
        min_horizon = model_out[horizon_col].min()
        max_horizon = model_out[horizon_col].max()
        self.wide_horizon_cols = [f"{horizon_col}{h}" for h in range(min_horizon, max_horizon + 1)]
        wide_model_out = (
            model_out
            .with_columns((horizon_col + pl.col(horizon_col).cast(str)).alias(horizon_col))
            .pivot(on=horizon_col, index = self.key_cols, values = self.obs_col)
        )
        self._build_train_X_y(min_horizon, max_horizon)
        
        transformed_model_out = (
            wide_model_out
            .group_by(self.key_cols)
            .map_groups(self._transform_one_group)
        )

        return transformed_model_out


    def _build_train_X_y(self, min_horizon, max_horizon):
        self.df = self.df.group_by(self.key_cols)
        for h in range(min_horizon, max_horizon + 1):
            if h < 0:
                shift_varname = self.time_col + "_m" + str(abs(h))
            else:
                shift_varname = self.time_col + "_p" + str(abs(h))
            
            if shift_varname not in self.shift_varnames:
                self.shift_varnames.append(shift_varname)
                self.df[shift_varname] = self.df[self.time_col].shift(-h)
        
        df_dropna = self.df.dropna()
        self.train_X = df_dropna[self.key_cols]
        self.train_y = df_dropna[self.shift_varnames]


    def _transform_one_group(self, wide_model_out):        
        templates = self.build_Shaake_templates(wide_model_out)
        transformed_model_out = self.apply_shuffle(wide_model_out[self.wide_horizon_cols], templates, self.value_cols)
        return transformed_model_out


    def _build_Shaake_templates(self, wide_model_out):
        weights = self.weighter.get_weights(self.train_X, wide_model_out[0, self.key_cols])
        selected_inds = np.random.choice(np.arange(self.train_y.shape[0]),
                                         size = wide_model_out.shape[0],
                                         replace = True,
                                         p = weights)
        templates = self.train_y[selected_inds]
        return templates
