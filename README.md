# National Bank of Ukraine IT Challenge

Hackathon from National Bank of Ukraine where we have to build machine learning model to predict USD/UAH exchange rate.

## Overview

### Input data

To solve this problem, we used data from open sources, including macroeconomic indicators and daily USD/UAH exchange rate.

<img src="https://github.com/imgremlin/Photos/blob/master/interp.png?raw=true" width="400px"> 

### Quality metric

We used Mean absolute error. This metric is pretty interpretable because it has the same unit of measurement as the initial series. 

<img src="https://github.com/imgremlin/Photos/blob/master/mae_nbu.png?raw=true" width="200px">

### Python libraries we used

<img src="https://github.com/imgremlin/Photos/blob/master/used_libs.jpeg?raw=true" width="200px">

## Model based on economical factors

These data fully describing the economic situation in Ukraine, but have some significant drawbacks (about it later).

<img src="https://github.com/imgremlin/Photos/blob/master/df_head.png?raw=true" width="200px">

Our pipeline contains few steps:
* Preprocessing
* Feature engineering
* Modelling

### Preprocessing

We used the monthly average rate since all parameters are measured each month/quarter/year.

Linear interpolation was used to fill in missing data, which are not measured monthly.

Data was used from 2015 to the 2020 year inclusive because before 2014-2015 years USD/UAH exchange rate was much lower than now, so it would be rather hard to use older data. (*)

### Feature engineering

In some columns you can see outliers, which negatively affects the performance of the model. They were detected and eliminated by the IQR method.

<img src="https://github.com/imgremlin/Photos/blob/master/before_iqr.png?raw=true" width="200px">

<img src="https://github.com/imgremlin/Photos/blob/master/after_iqr.png?raw=true" width="200px">

Also, we added lags for all columns for 1,2,3,6, and 12 months. We will tell more about lags in the time series section.

Choosing the columns that give the best result. We used our custom function based on backward feature selection to select the most useful columns for modeling because the boosting model is sensitive to the columns in the dataset.

Validation was performed on the basis of TimeSeriesSplit from the sklearn library to test how the model performs on the entire dataset. The diagram illustrates the principle of division into training and test parts.

<img src="https://github.com/imgremlin/Photos/blob/master/ts_split_validation.jpeg?raw=true" width="200px">

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





* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - *Dependency Management*
* [ROME](https://rometools.github.io/rome/) - Used to generate **RSS Feeds**
