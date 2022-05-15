from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi
import os


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class
    Parameters
    ----------
    filename: str
        Path to .npy data file
    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used
    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class
    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets
    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(os.path.join('datasets', f))

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        def update_loss(p,s,r):
            losses.append(p.loss(X,y))
        perceptron = Perceptron(callback=update_loss)
        perceptron.fit(X,y)

        # Plot figure of loss as function of fitting iteration
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=np.arange(len(losses)), y=losses, name= "Loss Value",
                line=dict(color='firebrick', width=2))
        )
        fig.update_layout(title = f"Loss as perceptron is updated - {n} case",
                        xaxis_title="Amount of Updates",
                        yaxis_title="Misclassification Error Loss")
        fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix
    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse
    cov: ndarray of shape (2,2)
        Covariance of Gaussian
    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black", showlegend=False)


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        data = np.load(os.path.join('datasets', f))
        X = data[:,:-1]
        y = data[:,-1].astype(int)


        # Fit models and predict over training set
        lda = LDA()
        gnb = GaussianNaiveBayes()
        lda.fit(X,y)
        gnb.fit(X,y)
        lda_pred = lda.predict(X)
        gnb_pred = gnb.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        symbols = np.array(["circle", "diamond", "square"])
        lda_accuracy = accuracy(y, lda_pred)
        gnb_accuracy = accuracy(y, gnb_pred)
        titles = [("LDA", lda_accuracy), ("Gaussian Naive Bayes", gnb_accuracy)]
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Results from %s Model. Accuracy: %.3f" % t for t in titles],
                    horizontal_spacing = 0.01, vertical_spacing=.03)
        fig.update_layout(title=f"Comparison Between LDA and GNB. Dataset: {f}", margin=dict(t=100))\
            .update_xaxes(visible=False). update_yaxes(visible=False)

        # Add traces for data-points setting symbols and colors
        for i, pred in enumerate([lda_pred, gnb_pred]):
            fig.add_traces([
                go.Scatter(x=X[:,0], y=X[:,1], mode="markers", showlegend=False,
                        marker=dict(color=pred, symbol=symbols[y], colorscale=[custom[0], custom[-1]], 
                        line=dict(color="black", width=1)) )],
                    rows=1, cols=i + 1
            )

        # Add `X` dots specifying fitted Gaussians' means
        for i, m in enumerate([lda, gnb]):
            fig.add_traces([
                go.Scatter(x=m.mu_[:,0], y=m.mu_[:,1], mode="markers", showlegend=False,
                        marker=dict(color="black", symbol="x", 
                        line=dict(color="black", width=1)) )],
                    rows=1, cols=i + 1
            )
        

        # Add ellipses depicting the covariances of the fitted Gaussians
        fig.add_traces([get_ellipse(lda.mu_[k,:], lda.cov_) for k in range(lda.mu_.shape[0])], rows=1, cols=1)
        fig.add_traces([get_ellipse(gnb.mu_[k,:], np.diag(gnb.vars_[k,:])) for k in range(gnb.mu_.shape[0])], rows=1, cols=2)
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()