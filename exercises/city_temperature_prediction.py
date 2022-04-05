from cProfile import label
from IMLearn.learners.regressors import linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    full_data = pd.read_csv(filename, parse_dates=["Date"]).dropna().drop_duplicates()

    #change date to the day in the year
    full_data = full_data.apply(lambda x : [obj.timetuple().tm_yday for obj in x] if x.name == "Date" else x)

    #delete samples with Temp< -10
    full_data = full_data.drop(full_data[full_data.Temp < -10].index)
    
    return full_data




if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data("/home/alonbentzi/IML.HUJI/datasets/City_Temperature.csv")
    
    # Question 2 - Exploring data for specific country
    Israel_data = data.loc[data['Country'] == "Israel"]

    #convert "YEAR" to string for px.scatter function
    Israel_data["Year"] =  Israel_data["Year"].astype(str)
    
    #plot Israel temp as function of Day of the year
    fig = px.scatter(Israel_data, x="Date", y="Temp", color="Year",
                 title="Temp as a function of Day of the year | Israel")
   
    fig.write_image(os.path.join("/home/alonbentzi/IML.HUJI/exercises/.plots", "Israel_data.png"))             

    # grouping by 'Month'
    grouped_by_month = data.groupby('Month').agg({'Temp' : np.std})

    fig = px.bar(grouped_by_month, x=np.arange(1,13), y="Temp",
                    labels={'x': "Month",
                            'Temp': "Temp (std)"},
                    title="std Temp as a function of Month")
                    
    fig.write_image(os.path.join("/home/alonbentzi/IML.HUJI/exercises/.plots", "Month_tmp.png"))   

    # Question 3 - Exploring differences between countries
    
    # grouping by 'Country & 'Month'
    grouped_by_month_and_country = data.groupby(['Month','Country']).Temp.agg([np.mean, np.std])
    grouped_by_month_and_country = grouped_by_month_and_country.reset_index('Country')
    print(grouped_by_month_and_country.shape)
    print(grouped_by_month_and_country.columns)

    fig = px.line(grouped_by_month_and_country, y='mean' ,color='Country',
                    error_y= 'std', 
                    labels={'x': "Month",
                            'Temp': "Temp (Avg)"},
                    title="std Temp as a function of Month")

    fig.write_image(os.path.join("/home/alonbentzi/IML.HUJI/exercises/.plots", "Month_tmp_with_err.png"))                  
   
    

    # Question 4 - Fitting model for different values of `k`
    train_x, train_y, test_x, test_y = split_train_test(Israel_data['Date'], Israel_data['Temp'])
    losses_array = np.empty((0,2), (int,float))
    for k in range(1,11):
        model = PolynomialFitting(k)
        model._fit(train_x, train_y)
        temp_loss =  round(model._loss(test_x, test_y), 2)
        losses_array = np.append(losses_array, [[k, temp_loss]], axis=0)
       
    
    fig = px.bar(losses_array, x=losses_array[:,0], y=losses_array[:,1],
                labels={'x': "K", "y": "Temp_loss"})  
    fig.write_image(os.path.join("/home/alonbentzi/IML.HUJI/exercises/.plots", "error_for_each_k.png"))


    # Question 5 - Evaluating fitted model on different countries
    
    BEST_K = 5
    counries = []
    loss_countries = []
    model_5 = PolynomialFitting(BEST_K)
    model_5.fit(Israel_data["Date"], Israel_data["Temp"])
    for country in data["Country"].unique():
        if country == "Israel": continue
        df_country = data[data["Country"] == country]
        loss = model_5.loss(df_country['Date'], df_country['Temp'])
        counries.append(country)
        loss_countries.append(loss)

    #convert arrays to np.array
    counries = np.array(counries)
    loss_countries = np.array(loss_countries)
    fig = px.bar(x=counries, y=loss_countries,
        labels={'x': "Countries", "y": "Temp_loss"})
    fig.write_image(os.path.join("/home/alonbentzi/IML.HUJI/exercises/.plots", "Q5.png"))    

