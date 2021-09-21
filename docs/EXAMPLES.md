<h1 align="center">User Guide</h1>

<p align="center"> tablespoon </p>


Data must have the following columns `y`(actuals) and `ds` (date). This is similar to how the prophet library
accepts data for forecasts.

# Python

```python
import numpy as np
import pandas as pd
from tablespoon import Naive, Mean, Snaive


# pull and clean data
df = (
    pd.read_csv("https://storage.googleapis.com/data_xvzf/m5_state_sales.csv")
    .query("state_id == 'CA'")
    .rename(columns={"date": "ds", "sales": "y"})
    .assign(y=lambda df: np.log(df.y))
)

# make lag of 1 default
m = Naive()
forecast = m.predict(df, horizon=10)
print(forecast.head())

m = Mean()
forecast = m.predict(df, horizon=10)
print(forecast.head())

# make lag of 7 default. weekly seasonality
m = Snaive()
forecast = m.predict(df, horizon=10)
print(forecast.head())
```
