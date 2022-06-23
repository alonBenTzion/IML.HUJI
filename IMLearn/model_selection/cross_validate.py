from __future__ import annotations
from copy import deepcopy
from random import sample
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator
from IMLearn.utils.utils import split_train_test    


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray,], float], cv: int = 5) -> Tuple[float, float]: #check about ... in the brackets of callable
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    #prepering the folds
    num_of_samples = X.shape[0]
    groups = np.tile(np.arange(cv), (num_of_samples // cv) + 1)
    #cut extra slots in groups
    groups_cut = groups[:num_of_samples]
    
    training_loss = []
    validation_loss = []
    #training the model
    for fold in range(cv):
        train_samples = X[groups_cut != fold]
        validation_samples = X[groups_cut == fold]
        train_labels = y[groups_cut != fold]
        validation_labels = y[groups_cut == fold]
        

        estimator.fit(train_samples, train_labels)
        validation_loss.append(scoring(validation_labels, estimator.predict(validation_samples)))
        training_loss.append(scoring(train_labels, estimator.predict(train_samples)))

    return np.mean(training_loss), np.mean(validation_loss)    
        
