# Extended Example

This example includes the following

1. adding midding dates
2. forward fill of `NA`
3. backward fill of `NA`
4. Using many forecast methods
5. making many plots

```python

import datetime as dt

import numpy as np
import pandas as pd
import pandas_datareader as pdr
import tablespoon as tbsp
from mizani.breaks import date_breaks
from plotnine import *

# Run if this is your first time installing cmdstanpy
# from cmdstanpy import install_cmdstan
# install_cmdstan()

# pull Apple open stock price
# columns must have the columns "ds" and "y"
# this time series is at the daily level. the seasonality is 7 days.
start_date = dt.datetime(2021, 8, 1)
end_date = dt.datetime(2022, 1, 1)
df = (pdr.get_data_yahoo("AAPL", start_date, end_date)
        .loc[:,['Open' ]]
        .reset_index()
        .rename(columns = {'Date': 'ds', 'Open': 'y'})
        .assign(y=lambda df: np.log(df.y))
    )

# It is common for time series data to have missing dates. 
# We need to fill in the missing values.
df_date_range = pd.DataFrame(pd.date_range(start=start_date, end=end_date), columns=['ds'])
df_complete_dates = df_date_range.merge(df, how='left', on='ds')
df_filled_forward = df_complete_dates.fillna(method = 'ffill') # fill NA with last valid value 
df_filled = df_filled_forward.fillna(method =  'bfill') # fill starting missing values with next valid value

# Snaive model
sn = tbsp.Snaive()
df_sn = (sn.predict(df_filled, horizon=7, frequency="D", lag = 7, uncertainty_samples = 8000).assign(model = 'snaive'))

# Complete Data is Required: Models Error when time series is missing dates 
n = tbsp.Naive()
df_n = (n.predict(df_filled, horizon=7*4, frequency="D", lag = 1, uncertainty_samples = 8000).assign(model = 'naive'))

# Complete Data is Required: Models Error when time series is missing dates 
m = tbsp.Mean()
df_m = (m.predict(df_filled, horizon=7*4, frequency="D", lag = 1, uncertainty_samples = 8000).assign(model = 'mean'))

# make some nice colors for plots
theme_set(theme_538)
palette = ["#000000", "#ee1d52"]

df_actuals_forecasts_sn = pd.concat([df_filled, df_sn])
p = (
    ggplot(df_actuals_forecasts_sn, aes(x="ds", y="y"))
    + geom_line(aes(y = 'y'), color = palette[0])
    + geom_point(aes(y = 'y_sim'), color = palette[1], size = 0.1, alpha = 0.1)
    + scale_x_datetime(breaks=date_breaks("1 month"))
    + theme(axis_text_x=element_text(angle=45))
    + xlab("")
    + ggtitle("Stock Price (Snaive)")
    + scale_color_manual(palette)
)
p.save(filename="forecasts_sn.jpg", width=14, height=3)

df_actuals_forecasts_n = pd.concat([df_filled, df_n])
p = (
    ggplot(df_actuals_forecasts_n, aes(x="ds", y="y"))
    + geom_line(aes(y = 'y'), color = palette[0])
    + geom_point(aes(y = 'y_sim'), color = palette[1], size = 0.1, alpha = 0.1)
    + scale_x_datetime(breaks=date_breaks("1 month"))
    + theme(axis_text_x=element_text(angle=45))
    + xlab("")
    + ggtitle("Stock Price (Naive)")
    + scale_color_manual(palette)
)
p.save(filename="forecasts_n.jpg", width=14, height=3)

df_actuals_forecasts_m = pd.concat([df_filled, df_m])
p = (
    ggplot(df_actuals_forecasts_m, aes(x="ds", y="y"))
    + geom_line(aes(y = 'y'), color = palette[0])
    + geom_point(aes(y = 'y_sim'), color = palette[1], size = 0.1, alpha = 0.1)
    + scale_x_datetime(breaks=date_breaks("1 month"))
    + theme(axis_text_x=element_text(angle=45))
    + xlab("")
    + ggtitle("Stock Price (Mean)")
    + scale_color_manual(palette)
)
p.save(filename="forecasts_m.jpg", width=14, height=3)


```