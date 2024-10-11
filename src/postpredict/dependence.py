import abc

from .util import argsort_random_tiebreak

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
        model_out: pandas dataframe with sample predictions that do not
        necessarily capture temporal dependence.
        
        Returns
        -------
        A copy of the model_out parameter, with sample indices updated so that
        they reflect the estimated temporal dependence structure.
        """
    
    
    def apply_shuffle(self, wide_model_out, value_cols, templates):
        """
        Given a collection of samples and an equally-sized collection of
        "dependence templates", shuffle the samples to match the rankings in the
        dependence templates.
        
        Parameters
        ----------
        wide_model_out: pandas dataframe with sample predictions that do not
        necessarily capture temporal dependence.
        value_cols: character vector of columns in `wide_model_out` that contain
        predicted values over time. These should be given in temporal order.
        templates: numpy array of shape (wide_model_out.shape[0], len(value_cols))
        containing
        
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

        sorted_wmo = wide_model_out.copy()
        for i, c in enumerate(value_cols):
            sorted_wmo[c] = sorted(wide_model_out[c])
            sorted_wmo.loc[col_orderings[c], c] = sorted_wmo[c].values
        
        return sorted_wmo
