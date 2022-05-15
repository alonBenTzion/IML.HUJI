import numpy as np
from typing import Tuple
from IMLearn.metalearners import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
from IMLearn.metrics.loss_functions import accuracy
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(lambda: DecisionStump(), n_learners)
    adaboost.fit(train_X, train_y)

    
    x_axis = np.arange(1, n_learners + 1)
    train_loss = []
    test_loss = []
    
    for t in x_axis:
        train_loss.append(adaboost.partial_loss(train_X, train_y, t))
        test_loss.append(adaboost.partial_loss(test_X, test_y, t))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_axis, y=train_loss, line=dict(color='royalblue', width=3), name="Train Loss"))
    fig.add_trace(go.Scatter(x=x_axis, y=test_loss, line=dict(color='firebrick', width=3), name="Test Loss"))
    fig.update_layout(title = f"Test & Train Sets Loss as a function of Number of Weak Learners - Noise Factor {noise}",
                        xaxis_title="# Weak Learners",
                        yaxis_title="MSE")
    fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig = make_subplots(rows=2, cols=2, subplot_titles=[f"{t} Weak Learners" for t in T],
                    horizontal_spacing = 0.02, vertical_spacing=.04)
    symbols = np.array(["x", "circle", "x"])
    for i, t in enumerate(T):
        pred = lambda X: adaboost.partial_predict(X,t)
        fig.add_traces([decision_surface(pred, lims[0], lims[1], showscale=False),
                    go.Scatter(x=test_X[:,0], y=test_X[:,1], mode="markers", showlegend=False,
                               marker=dict(color=test_y.astype(int), symbol=symbols[test_y.astype(int)], colorscale=[custom[0], custom[-1]], 
                                           line=dict(color="black", width=1)) )], 
                   rows=(i//2) + 1, cols=(i%2)+1)

    fig.update_layout(title=f"Decision boundaries for changing numbers of weak learners  - Noise Factor {noise}", margin=dict(t=100))\
    .update_xaxes(visible=False). update_yaxes(visible=False)

    fig.show()


    # Question 3: Decision surface of best performing ensemble
    
    predict = lambda X: adaboost.partial_predict(X, best_t)
    best_t = np.argmin(test_loss) + 1
    test_prediction = adaboost.partial_predict(test_X, best_t)
    fig = go.Figure()
    fig.add_traces([decision_surface(predict, lims[0], lims[1], showscale=False),
                    go.Scatter(x=test_X[:,0], y=test_X[:,1], mode="markers", showlegend=False,
                               marker=dict(color=test_y.astype(int), symbol=symbols[test_y.astype(int)], colorscale=[custom[0], custom[-1]], 
                                           line=dict(color="black", width=1)) )])
    accur = accuracy(test_y, test_prediction)
    fig.update_layout(title=f"Decision Boundaries for Ensamble Size {best_t} - Noise Factor {noise}. Accuracy: {accur}", margin=dict(t=100))\
        .update_xaxes(visible=False). update_yaxes(visible=False)
    
    fig.show()

    # Question 4: Decision surface with weighted samples
    factor_size = (adaboost.D_ / np.max(adaboost.D_)) * 5
    fig = go.Figure()
    fig.add_traces([decision_surface(adaboost.predict, lims[0], lims[1], showscale=False),
                    go.Scatter(x=train_X[:,0], y=train_X[:,1], mode="markers", showlegend=False,
                        marker=dict(size = factor_size, color=train_y.astype(int), symbol=symbols[train_y.astype(int)], colorscale=[custom[0], custom[-1]], 
                        line=dict(color="black", width=1)) )])
    fig.update_layout(title=f"Train Samples As A Function Of Their Weight In The Distribution - Noise Factor {noise}.", margin=dict(t=100))\
       . update_yaxes(visible=False) .update_xaxes(visible=False)
    
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)

