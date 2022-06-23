from matplotlib.pyplot import flag
import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test
from sklearn.metrics import roc_curve
from IMLearn.model_selection import  cross_validate
from IMLearn.metrics import mean_square_error


import plotly.graph_objects as go
def plot_convergence_rate(y:np.ndarray, title:str):
    fig = go.Figure([go.Scatter(x=list(range(len(y))),
                                y=y, 
                                mode="markers+lines",
                                showlegend=True,
                                marker=dict(color="black", opacity=.7),
                                line=dict(color="black",
                                width=1))],
                                layout=go.Layout(title=title ,
                                                xaxis={"title": "x - Iterations"},   
                                                yaxis={"title": "y - Value"},
                                                height=400))
    # fig.show()     
    fig.write_image(f"exercises/.plots/ex6/{title}.png")                                       


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))



def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    weights_arr, values_arr = [], []

    def func(weights, val, **kwargs):
        weights_arr.append(weights)
        values_arr.append(val)

    return func, weights_arr, values_arr



def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    for module in [[L1, "L1"], [L2, "L2"]]:
        for eta in etas:
            values = []
            weights_ls = []

            def recorder_func(solver: GradientDescent,
                               weights: np.ndarray,
                               val: np.ndarray,
                               grad: np.ndarray,
                               t: int,
                               eta: float,
                               delta: float):
                values.append(val)
                weights_ls.append(weights)
            m = module[0](init)          
            GradientDescent(FixedLR(eta),callback=recorder_func).fit(f=m,X=None,y=None)
            #plotting the trajectory
            plot_descent_path(module[0], np.asarray(weights_ls),
                             title=f"{module[1]}, lr = {eta} - descent trajectory").write_image(f"exercises/.plots/ex6/traj_{module[1]}_{eta}.png")
            #plotting the convergence rate
            plot_convergence_rate(values, title=f"{module[1]} convergence rate with fixed lr = {eta}")

    

def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    for gama in gammas:
        values = []
        weights_ls = []

        def recorder_func(solver: GradientDescent,
                            weights: np.ndarray,
                            val: np.ndarray,
                            grad: np.ndarray,
                            t: int,
                            eta: float,
                            delta: float):
            values.append(val)
            weights_ls.append(weights)
        
        GradientDescent(ExponentialLR(eta, gama), callback=recorder_func).fit(L1(init), None, None) 
        # Plot algorithm's convergence for the different values of gamma
        plot_convergence_rate(values, title=f"L1 convergence rate with gamma = {gama}")
        # Plot descent path for gamma=0.95
        if gama == 0.95:
             plot_descent_path(L1, np.asarray(weights_ls), f'gama {gama}').write_image(f"exercises/.plots/ex6/traj_exp_decay_0.95.png")

    
    
    
  
   
    


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    lg = LogisticRegression().fit(X=X_train, y=y_train)
    fpr, tpr, thresholds = roc_curve(y_train, lg.predict_proba(X_train.to_numpy()))
    fig = go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                         line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds,
                         name="", showlegend=False, marker_size=5,
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model}}$",
                        xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                        yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$")))
    #fig.show()
    fig.write_image("exercises/.plots/ex6/lgr_roc_curve.png")

    
    
    alpha_hat = thresholds[np.argmax(tpr - fpr)]
    print(f"best alpha value: {alpha_hat}")

    # Plotting convergence rate of logistic regression over SA heart disease data
    

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    lambdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    cv = lambda reg, lam :  cross_validate(LogisticRegression(penalty=reg, alpha=0.5, lam=lam), X_train.to_numpy(),
                                        np.array(y_train), mean_square_error)
    for regulator in ['l1', 'l2']:
        lambdas_scores = [cv(regulator, lam)[1] for lam in lambdas]
        best_lambda = lambdas[np.argmin(lambdas_scores)]
        loss = LogisticRegression(penalty=regulator, alpha=0.5, lam=best_lambda).fit(X_train, y_train).loss(X_test, y_test)
        print(f'{regulator} : lambda {best_lambda}, loss {loss}')


if __name__ == '__main__':
    np.random.seed(0)
    # compare_fixed_learning_rates()
    # compare_exponential_decay_rates()
    fit_logistic_regression()
