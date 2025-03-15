# Instagram Reach Analysis & Forecasting

## Overview
Instagram Reach Forecasting is the process of predicting the number of people that an Instagram post, story, or other content will reach based on historical data and various influencing factors.

For content creators and professionals using Instagram, predicting reach is valuable for optimizing their social media strategy. By analyzing past performance, I can make informed decisions on when to post, what type of content to create, and how to improve audience engagement.

---

## Dataset
The dataset used in this project contains Instagram reach data along with dates. Below is a sample of the [dataset](https://statso.io/social-media-reach-forecasting-case-study/):

| Date       | Instagram Reach |
|------------|----------------|
| 2022-04-01 | 7620           |
| 2022-04-02 | 12859          |
| 2022-04-03 | 16008          |
| 2022-04-04 | 24349          |
| 2022-04-05 | 20532          |

---

## Data Preprocessing
I convert the `Date` column into a datetime format to facilitate time series analysis:

```python
import pandas as pd

data['Date'] = pd.to_datetime(data['Date'])
print(data.head())
```

---

## Exploratory Data Analysis
### Instagram Reach Over Time
To analyze the trend of Instagram reach over time, I use a line chart:

![Instagram Reach Over Time](https://github.com/Sourabh1710/Instagram-Reach-Forecasting/blob/main/images/Instagram%20Reach%20Trend.png)

### Instagram Reach by Day
To analyze reach for each day, I use a bar chart:

![Instagram Reach by Day](https://github.com/Sourabh1710/Instagram-Reach-Forecasting/blob/main/images/INstagram%20Reach%20by%20Day.png)

### Distribution of Instagram Reach
A box plot is used to analyze the distribution of Instagram reach:

![Distribution of Instagram Reach](https://github.com/Sourabh1710/Instagram-Reach-Forecasting/blob/main/images/Instagram%20Reach%20Box%20Plot.png)

---

## Instagram Reach Based on Days of the Week
I create a new column `Day` using `dt.day_name()` to extract the day of the week from the `Date` column:

```python
data['Day'] = data['Date'].dt.day_name()
print(data.head())
```

I then compute statistics (mean, median, standard deviation) for each day:

```python
import numpy as np

day_stats = data.groupby('Day')['Instagram reach'].agg(['mean', 'median', 'std']).reset_index()
print(day_stats)
```

| Day       | Mean Reach | Median Reach | Std Dev |
|-----------|-----------|--------------|----------|
| Friday    | 46666.85  | 35574.0      | 29856.94 |
| Monday    | 52621.69  | 46853.0      | 32296.07 |
| Saturday  | 47374.75  | 40012.0      | 27667.04 |
| Sunday    | 53114.17  | 47797.0      | 30906.16 |
| Thursday  | 48570.92  | 39150.0      | 28623.22 |
| Tuesday   | 54030.56  | 48786.0      | 32503.72 |
| Wednesday | 51017.27  | 42320.5      | 29047.87 |

### Visualization of Instagram Reach for Each Day

![Instagram Reach by Day of Week](https://github.com/Sourabh1710/Instagram-Reach-Forecasting/blob/main/images/Instagram%20Reach%20by%20Day%20of%20the%20Week.png)

---

## Time Series Forecasting for Instagram Reach
### Trend and Seasonality Analysis
To analyze the trend and seasonal patterns in Instagram reach:

![Trend and Seasonality](https://github.com/Sourabh1710/Instagram-Reach-Forecasting/blob/main/images/Instagram%20reach.png)

Since reach is affected by seasonality, I use the **SARIMA model** for forecasting.

### Determining Model Parameters (p, d, q)
To determine `p`, `d`, and `q` values, I use autocorrelation and partial autocorrelation plots:

#### Autocorrelation Plot

```python
pd.plotting.autocorrelation_plot(data["Instagram reach"])
```

![Autocorrelation Plot](https://github.com/Sourabh1710/Instagram-Reach-Forecasting/blob/main/images/Instagram%20reach%20Autocorrelation.png)

#### Partial Autocorrelation Plot

```python
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(data["Instagram reach"], lags=100)
```

![Partial Autocorrelation Plot](https://github.com/Sourabh1710/Instagram-Reach-Forecasting/blob/main/images/Instagram%20reach%20Partial%20Autocorrelation.png)

### Training SARIMA Model
Using the determined values (`p=8, d=1, q=2`), I train a SARIMA model:

```python
import statsmodels.api as sm
import warnings

p, d, q = 8, 1, 2
model = sm.tsa.statespace.SARIMAX(
    data['Instagram reach'], 
    order=(p, d, q), 
    seasonal_order=(p, d, q, 12)
)
model = model.fit()
print(model.summary())
```

### Forecasting Instagram Reach
After training, I make predictions and visualize the forecast:

![Instagram Reach Forecast](https://github.com/Sourabh1710/Instagram-Reach-Forecasting/blob/main/images/Instagram%20Reach%20Time%20Series%20and%20Predictions.png)

---

## Conclusion
Instagram reach forecasting is essential for optimizing social media strategies. By leveraging time series forecasting, I can predict reach based on historical data, helping content creators maximize their audience engagement.

---


## Author
Sourabh Sonker <br>
Data Scientist
