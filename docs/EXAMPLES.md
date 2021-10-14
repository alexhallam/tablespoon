<h1 align="center">User Guide</h1>

<p align="center"> tablespoon </p>


Data must have the following columns `y`(actuals) and `ds` (date). This is similar to how the prophet library
accepts data for forecasts.

# Python

```python
import numpy as np
import pandas as pd
from tablespoon import Naive, Mean, Snaive


# Pull and clean data
# This time series is at the daily level. The seasonality is 7 days
df = (
    pd.read_csv("https://storage.googleapis.com/data_xvzf/m5_state_sales.csv")
    .query("state_id == 'CA'")
    .rename(columns={"date": "ds", "sales": "y"})
    .assign(y=lambda df: np.log(df.y))
)

m = Naive()
forecast = m.predict(df, horizon=10, frequency="D", lag=1)
print(forecast.head())

m = Mean()
forecast = m.predict(df, horizon=10, frequency="D", lag=1)
print(forecast.head())

# lag of 7 (weekly seasonality)
m = Snaive()
forecast = m.predict(df, horizon=10, frequency="D", lag=7)
print(forecast.head())
```
