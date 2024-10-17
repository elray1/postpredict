import numpy as np


class EqualWeighter():
    def __init__(self) -> None:
        pass
    
    
    def get_weights(self, train_X, test_X):
        """
        Compute training set observation weights.
        
        Parameters
        ----------
        train_X: dataframe or array of shape (n_train, p)
            Training set features used for weighting. There is one row for each
            training set instance and one column for each of the p features.
        test_X: dataframe or array of shape (n_test, p)
            Test set features used for weighting. There is one row and one
            column for each of the p features.
        
        Returns
        -------
        numpy array of length n_train with weights for each training set
        instance, where weights sum to 1.
        """
        n_train = train_X.shape[0]
        n_test = test_X.shape[0]
        return np.full((n_test, n_train), 1 / n_train)
