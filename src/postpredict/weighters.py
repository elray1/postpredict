import numpy as np


class EqualWeighter():
    def __init__(self) -> None:
        pass
    
    
    def get_weights(X_train, X_test):
        """
        Compute training set observation weights.
        
        Parameters
        ----------
        X_train: dataframe or array of shape (n_train, p)
            Training set features used for weighting. There is one row for each
            training set instance and one column for each of the p features.
        X_test: dataframe or array of shape (1, p)
            Test set features used for weighting. There is one row and one
            column for each of the p features.
        
        Returns
        -------
        numpy array of length n_train with weights for each training set
        instance, where weights sum to 1.
        """
        n_train = X_train.shape[0]
        return np.full(n_train, 1 / n_train)
