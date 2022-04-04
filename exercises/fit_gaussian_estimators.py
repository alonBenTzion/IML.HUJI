from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"

NUM_OF_S = 1000


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu_, var_, = 10, 1
    S = np.random.normal(mu_, var_, NUM_OF_S)
    estimator = UnivariateGaussian()
    estimator.fit(S)
    print(f"{estimator.mu_, estimator.var_}")

    # Question 2 - Empirically showing sample mean is consistent
    def temp_expectation(index):
        return UnivariateGaussian().fit(S[:index]).mu_

    sizes = np.arange(10, 1010, 10)
    exp_val_differ = [abs(temp_expectation(size) - mu_)
                      for size in sizes]
    go.Figure([go.Scatter(x=sizes, y=exp_val_differ, mode='markers+lines',
                          name=r'$\widehat\Q2$')],
              layout=go.Layout(
                  title=r"$\text{Estimation of Expectation As Function "
                        r"Of Number Of Samples}$",
                  xaxis_title="$\\text{Number of samples}$",
                  yaxis_title="$\\text{Estimation differ}$",
                  height=500)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    go.Figure([go.Scatter(x=S, y=estimator.pdf(S), mode='markers',
                          name=r'$\widehat\Q3$')],
              layout=go.Layout(
                  title=r"$\text{PDF As Function Of Samples}$",
                  xaxis_title="$\\text{Samples}$",
                  yaxis_title="$\\text{PDF}$",
                  height=500)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu_ = np.array([0, 0, 4, 0])
    cov_ = np.array([[1, 0.2, 0, 0.5],
                     [0.2, 2, 0, 0],
                     [0, 0, 1, 0],
                     [0.5, 0, 0, 1]])
    S = np.random.multivariate_normal(mu_, cov_, NUM_OF_S)
    estimator = MultivariateGaussian().fit(S)
    print("Mu: ", estimator.mu_, "\n")
    print("Cov: ", estimator.cov_, "\n")

    # # Question 5 - Likelihood evaluation
    values = np.linspace(-10, 10, 200)
    mu_array = np.array(np.meshgrid(values, 0, values, 0)).T.reshape(-1, 4)
    log_likelihood = np.apply_along_axis(
        lambda x: MultivariateGaussian.log_likelihood(x, cov_, S),
        1,
        mu_array)
    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=values, y=values,
                             z=log_likelihood.reshape(200, 200).T,
                             colorbar=dict(title="Log Likelihood")))
    fig.update_layout(title=r"$\\text{Log Likelihood of samples with mean ["
                            r"0,f1,0,f3]}$",
                      xaxis_title="$\\text{f3 values}$",
                      yaxis_title="$\\text{f1 values}$",
                      height=500)
    fig.show()

    # Question 6 - Maximum likelihood
    print("Maximizing Values:")
    print("f1: %.3f " % mu_array[np.argmax(log_likelihood)][0],
          "f3: %.3f" % mu_array[np.argmax(log_likelihood)][2])


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()

