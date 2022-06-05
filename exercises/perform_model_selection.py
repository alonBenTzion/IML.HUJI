from __future__ import annotations
from email.policy import default
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    epsilon = np.random.normal(0, noise, n_samples)
    x_values = pd.DataFrame({"x": np.random.uniform(-1.2, 2, n_samples)})
    f = lambda x : (x+3)*(x+2)*(x+1)*(x-1)*(x-2)
    # vfunc = np.vectorize(f)
    # y_values = vfunc(x_values) + epsilon
    y_values = pd.Series(f(x_values["x"]) + epsilon, name="y")
    
    train_x, train_y, test_x, test_y = split_train_test(x_values, y_values, 2/3)
    # train_x, test_x=np.array(train_x[0]), np.array(test_x[0])
    # train_y,test_y=np.array(train_y), np.array(test_y)
    sorted_x = x_values["x"].sort_values()
    fig = go.Figure([
                    go.Scatter(x=sorted_x,
                                y=f(sorted_x),
                                mode="markers",
                                name="True Values",
                                line=dict(dash="dash"),
                                marker=dict(color="green", opacity=.7)),
                    go.Scatter(x=train_x,
                                y= train_y,
                                mode="markers",
                                 name="Train Set",
                                line=dict(color="blue"),
                                showlegend=True),
                    go.Scatter(x=test_x,
                                y=test_y,
                                mode="markers",
                                 name="Test Set",
                                line=dict(color="red"),
                                showlegend=True)
                     ])
    fig.update_layout(
        title=f"Train and test sets compared to true Model. {n_samples} samples with noise {noise}",
        xaxis_title="x",
        yaxis_title="y")
    fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_loss_array = []
    validation_loss_array = []
    for k in range(11):    
        train_loss, validation_loss = cross_validate(PolynomialFitting(k),train_x, train_y, mean_square_error)
        train_loss_array.append(train_loss), validation_loss_array.append(validation_loss)

    k_values = np.arange(0,11,1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=k_values, y=train_loss_array, line=dict(color='royalblue', width=3), name="Train Loss"))
    fig.add_trace(go.Scatter(x=k_values, y=validation_loss_array, line=dict(color='firebrick', width=3), name="Validation Loss"))
    fig.update_layout(title = f"Train & Validation Loss as a function of polynomial degree",
                        xaxis_title="# polynomial degree - K",
                        yaxis_title="Loss")
    fig.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_k = np.argmin(validation_loss_array)
    model = PolynomialFitting(best_k).fit(train_x, train_y)
    loss = np.round(mean_square_error(model.predict(test_x), test_y), 2)
    print(best_k, loss)




def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    raise NotImplementedError()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    raise NotImplementedError()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    # select_polynomial_degree(default, 0)
    # select_polynomial_degree(1500, 10)
