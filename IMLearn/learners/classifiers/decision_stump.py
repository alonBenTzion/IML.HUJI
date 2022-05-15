from __future__ import annotations
from ast import arg
from operator import mod
from re import S
from typing import Tuple, NoReturn
from setuptools import SetuptoolsDeprecationWarning
from IMLearn.base.base_estimator import BaseEstimator
import numpy as np
from itertools import product




class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None
        

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        def better_sign(feature: np.ndarray) -> np.ndarray:
            """
            return [loss, threshold, sign]
            """
            options = np.array([self._find_threshold(feature,y,-1),self._find_threshold(feature,y,1)])
            better_threshold_index = np.argmin(options,axis=0)[1]
            return np.array([options[better_threshold_index,1], options[better_threshold_index,0], (better_threshold_index * 2 - 1)])
            

        optional_thresholds = np.apply_along_axis(better_sign,0,X)
        self.j_ = np.argmin(optional_thresholds,axis=1)[0]
        self.threshold_, self.sign_ = optional_thresholds[1, self.j_], optional_thresholds[2, self.j_]
   

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
       
        prediction = (X[:,self.j_]>=self.threshold_).astype(int)
        prediction = prediction * 2 - 1
        return prediction * self.sign_

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        # The method I implemented has few steps:
        # First, sort the values, and than the labels respectively
        # then create an array contains in each slot the number of labeling errors if the i'th value was the threshold  
        sorted_indexes = values.argsort()
        values = values[sorted_indexes]
        labels = labels[sorted_indexes]
        weights = np.abs(labels)
        a = np.cumsum((labels * sign >= 0)*weights)
        a = np.roll(a,1)
        a[0] = 0
        b = np.cumsum(((labels * sign < 0)*weights)[::-1])[::-1]
        loss = a + b
        threshold_value_index = np.argmin(loss)
        if threshold_value_index == 0:
            threshold_value = np.inf * sign
        else:
            threshold_value = values[threshold_value_index]

        return (threshold_value, loss[threshold_value_index])

          
    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        indicator = (y*self._predict(X) < 0).astype(int)
        abs_y = np.abs(y)
        return np.sum(indicator*abs_y) / np.sum(abs_y)
        

    

   