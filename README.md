# National Bank of Ukraine IT Challenge

Hackathon from National Bank of Ukraine where we have to build machine learning model to predict USD/UAH exchange rate.

## Overview

### Input data

To solve this problem, we used data from open sources, including macroeconomic indicators and daily USD/UAH exchange rate.

<img src="https://github.com/imgremlin/Photos/blob/master/interp.png?raw=true" width="300px"> 

### Quality metric

We used Mean absolute error. This metric is pretty interpretable because it has the same unit of measurement as the initial series. 

<img src="https://github.com/imgremlin/Photos/blob/master/mae_nbu.png?raw=true" width="300px">

### Python libraries we used

<img src="https://github.com/imgremlin/Photos/blob/master/used_libs.jpeg?raw=true" width="500px">

## Model based on economical factors

These data fully describing the economic situation in Ukraine, but have some significant drawbacks (about it later).

<img src="https://github.com/imgremlin/Photos/blob/master/df_head.png?raw=true" width="1100px">

Our pipeline contains few steps:
* Preprocessing
* Feature engineering
* Modelling

### Preprocessing

We used the monthly average rate since all parameters are measured each month/quarter/year.

Linear interpolation was used to fill in missing data, which are not measured monthly.

Data was used from 2015 to the 2020 year inclusive because before 2014-2015 years USD/UAH exchange rate was much lower than now, so it would be rather hard to use older data. (*)

In some columns you can see outliers, which negatively affects the performance of the model. They were detected and eliminated by the IQR method.

<img src="https://github.com/imgremlin/Photos/blob/master/before_iqr.png?raw=true" width="400px">
<img src="https://github.com/imgremlin/Photos/blob/master/after_iqr.png?raw=true" width="400px">

### Feature engineering

We added lags for all columns for 1,2,3,6, and 12 months. We will tell more about lags in the time series section.

Choosing the columns that give the best result. We used our custom function based on backward feature selection to select the most useful columns for modeling because the boosting model is sensitive to the columns in the dataset.

Validation was performed on the basis of TimeSeriesSplit from the sklearn library to test how the model performs on the entire dataset. The diagram illustrates the principle of division into training and test parts.

<img src="https://github.com/imgremlin/Photos/blob/master/ts_split_validation.jpeg?raw=true" width="650px">

### Modeling

The selected model was XGBoost. We also tested other models such as LightGBM, CatBoost, RandomForests, KNN, and others. After selecting the model, the best parameters of the model were selected according to the cross-validation.
Hyperparameters of our model:

```
XGBRegressor(learning_rate=0.3, n_estimators=100, max_depth=9, subsample=0.9, colsample_bytree=0.7)
```

* learning_rate - step size shrinkage used in update to prevents overfitting
* n_estimators - number of trees in the model
* max_depth - maximum depth of a tree
* subsample - subsample ratio of the training instances
* colsample_bytree - the subsample ratio of columns when constructing each tree

### Conclusion

So why did we reject this approach? This decision had to be made due to a number of problems:

* Most parameters are calculated monthly or quarterly, and our goal is the exchange rate daily. Any kind of interpolation could not fully reflect the trends of the parameters for each day
* Unfortunately, the economic situation in the country is not stable enough, which causes many anomalous values that interfere with the model. We could eliminate them, however, these are real indicators of the country. If all these values are removed, a completely different economic picture will emerge.

As a result, we have a set of data that is difficult to interpret qualitatively for each day and which contains a fairly large amount of missing data through the calculation algorithms.

## Model based only on time series

```
A time series is a series of data points indexed (or listed or graphed) in time order.
```

That is, the data are organized by relatively deterministic timestamps and may, compared to random sample data, contain additional information that we can analyze. In our case, such series is the USD/UAH exchange rate.

<img src="https://github.com/imgremlin/Photos/blob/master/ts_plot.png?raw=true" width="400px">

But, as we saw during the solution of the problem, models for time series (ARIMA / SARIMA) are not optimal in this case due to the lack of certain patterns in the time series of our exchange rate. In addition, they are not profitable from the point of view of launching into production: they require more time for data preparation, require frequent retraining, and are quite difficult to configure.

So we generated a bunch of features from the existing time series and built a regression model.

Generated features:
* Time-series lags
* Statistics for a certain period (minimum / maximum, average, variance)
* Trends direction
* The rapidity of time series change
* Testing the hypothesis of stationarity of intervals

### Time-series lags

To confirm the hypothesis of the possibility of creating this feature, we will construct an autocorrelation plot to see whether the time series lags correlate with each other because they will play a leading role in the future model.

<img src="https://github.com/imgremlin/Photos/blob/master/autocorrelation.png?raw=true" width="400px">

Confidence intervals are depicted as a cone. The 95% confidence interval suggests that the correlation value outside this cone is a correlation, not a statistical error.

<img src="https://github.com/imgremlin/Photos/blob/master/lags.png?raw=true" width="450px">

### Statistics for a certain period

To get more information about our time series, we need certain statistical information on different intervals. We will be able to improve the model by providing additional data about the series. However, we do not forget that it is necessary to calculate statistics on some lag of a series (instead of on a series) to avoid data leakage.

<img src="https://github.com/imgremlin/Photos/blob/master/statistics_4_analysis.png?raw=true" width="450px">

### Time-series stationarity

A stationary time series is one whose properties, namely the mean and variance, do not depend on the time at which the series is observed. Thus, time series with trends, or with seasonality, are not stationary — the trend and seasonality will affect the value of the time series at different times. On the other hand, a white noise series is stationary — it does not matter when you observe it, it should look much the same at any point in time.

Why is stationary so important? Because it is easier to make predictions on stationary series because we can assume that future statistical properties will not differ from those currently observed. Therefore, we test the hypothesis of stationarity (Dickey-Fuller test) of individual time intervals and add this information to the model.

Dickey-Fuller test:
* Null hypothesis: the time series is nonstationary (has some time-dependent structure)
* Alternative hypothesis: (rejecting the null hypothesis) the time series is stationary
* We interpret this result using the p-values from the test. A value of p below the threshold level of 5% suggests that we reject the null hypothesis, otherwise we can assume a null hypothesis

<img src="https://github.com/imgremlin/Photos/blob/master/stationary.png?raw=true" width="450px">

### Improvement of this method
For more accurate forecasting for long periods (more than 14 days), we have developed a model with additional training at certain stages. The idea is as follows:
* Divide the interval into which we want to make predictions into several smaller intervals
* Using training data predict the first small interval and add our predictions as true labels to the training set
* Retrain model with a bigger training dataset and predict next small interval
* Repeat these steps for all small intervals

This technique is called pseudo-labeling and used in semi-supervised learning.

(\*) we could use the exchange rate difference between this day and the next day as a target for our model to avoid this problem and to make target distribution more stationary
