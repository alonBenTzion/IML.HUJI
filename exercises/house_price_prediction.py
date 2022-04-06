# from statistics import linear_regression
from re import X
from turtle import title

from scipy import stats
from pyparsing import PrecededBy
from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

import os
from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """

    #load all data
    df = pd.read_csv(filename).dropna().drop_duplicates()
    # preproccesing the data:
    

    #irrelevant featurs
    df = df.drop(columns=["zipcode", "id", "date", "lat", "long"])

    # values range according to Kaggle site
    df = df[(df.view >= 0) & (df.view <= 4)]
    df = df[(df.grade >= 1) & (df.grade <= 13)]
    df = df[(df.condition >= 1) & (df.condition <= 5)]
    df = df[df.yr_built >= 1900]
    df = df[df.price > 0]
    df = df[df.sqft_living > 0]
    df = df[df.sqft_living >= df.sqft_above]
    #remove outliners
    df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]

    labels = df["price"]
    features = df.drop(columns="price")
    
    return features, labels
    



def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against
 
   output_path: str (default ".")
        Path to folder in which plots are saved
    """
    
    #create plots
    for i, feature in  enumerate(X):
        corr = np.cov(X[feature], y)[0,1] / (np.std(X[feature]) * np.std(y))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x= X[feature], y= y, mode="markers"))
        fig.update_xaxes(title_text=feature)
        fig.update_yaxes(title_text="price")
        fig.update_layout(title_text=f"Correlation value: {corr}")
        fig.write_image(os.path.join(output_path, "price_as_func_of_%s.png"% feature)) 

      


        


if __name__ == '__main__':
    np.random.seed(0)
    
    # Question 1 - load data
    featurs, labels = load_data("/home/alonbentzi/IML.HUJI/datasets/house_prices.csv")
    
    
    
  
    # Question 2 - Feature evaluation with respect to response
    # feature_evaluation(featurs, labels, "/home/alonbentzi/IML.HUJI/exercises/.plots")
    
    # Question 3 - Split samples into training- and testing sets.
    train_x, train_y, test_x, test_y = split_train_test(featurs, labels)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)


    percentages = np.arange(10,101)
    mean_loss = []
    std_loss = []
    for percent in range(10, 101, 1):
        
        temp_losses = np.empty((10,))
        
        for experiment_number in range(10):
            
            sub_train_x = train_x.sample(frac=percent/100, axis=0)  
            sub_train_y = train_y.loc[sub_train_x.index]
            model = LinearRegression()
            model._fit(sub_train_x, sub_train_y)

            loss = model._loss(test_x, test_y)
            temp_losses[experiment_number] = loss

        mean_loss.append(temp_losses.mean())
        std_loss.append(temp_losses.std()) 

    #convert arrays to np arrays for plotly
    mean_loss = np.array(mean_loss)
    std_loss = np.array(std_loss)

    # plot average loss as function of training size with error ribbon of size(mean-2*std, mean+2*std)  
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=percentages, y=mean_loss, mode="markers+lines", name="Mean Loss", line=dict(dash="dash"), marker=dict(color="green", opacity=.7)))
    fig.add_trace(go.Scatter(x=percentages, y=mean_loss-2*std_loss, fill=None, mode="lines", line=dict(color="lightgrey"), showlegend=False))
    fig.add_trace(go.Scatter(x=percentages, y=mean_loss+2*std_loss, fill='tonexty', mode="lines", line=dict(color="lightgrey"), showlegend=False))
    fig.update_layout(go.Layout(
            title_text="Average loss as a function of Sampels percentage with ribbon of (+,-) 2*std",
            xaxis={"title": "Train set percentage"},
            yaxis={"title": "MSE Loss"}))
    fig.write_image(os.path.join("/home/alonbentzi/IML.HUJI/exercises/.plots", "avg_loss_as_f_of_S_percent.png"))                     





