# Skill Score

One of the primary purposes of this package is to generate forecast methods as benchmarks.

One way to use these methods as benchmarks is to generate *skill scores*.

Skill scores have the general form:

$$
\frac{\text{CRPS}_{\text{Naïve}} - \text{CRPS}_{\text{Alternative}}}{\text{CRPS}_{\text{Naïve}}}.
$$

In the equation above we used `CRPS`, but any metric may be used.


## Example

```python
import pandas as pd
import tablespoon as tbsp
from tablespoon.data import APPL
import numpy as np
import datetime as dt


train = APPL.query('ds <= "2021-11-01"')
test = APPL.query('ds > "2021-11-01"')


def crps(scalar_y, vec_of_forecast):
    x = np.sort(vec_of_forecast)
    m = len(vec_of_forecast)
    return (2 / m) * np.mean((x - scalar_y) * (m * np.where(scalar_y < x, 1, 0) - np.arange(start=0, stop=m, step=1) + 1 / 2))

# make naive forecast
df_n = tbsp.Naive().predict(
    train, horizon=7 * 4, frequency="D", lag=1, uncertainty_samples=8000
).assign(model="naive").assign(ds = lambda df: df.ds.dt.strftime('%Y-%m-%d'))

# make alternative forecast
df_m = tbsp.Mean().predict(
    train, horizon=7 * 4, frequency="D", lag=1, uncertainty_samples=8000
).assign(model="mean").assign(ds = lambda df: df.ds.dt.strftime('%Y-%m-%d'))

# join forecasts together and left join with actuals
df_forecast_actual_n = test.merge(df_n, how='left', on = 'ds')
df_forecast_actual_m = test.merge(df_m, how='left', on = 'ds')
df_forecasts = pd.concat([df_forecast_actual_n, df_forecast_actual_m], axis=0)

# calculate crps by date and model
df_crps = (df_forecasts
            .groupby(by = ['model', 'ds'], as_index=False)
            .apply(lambda df: crps(df["y"].iat[0], df["y_sim"]))
            .assign(metric="crps")
            .rename(columns={None: "value"}))

# get mean crps
df_mean_crps = df_crps.groupby(by = ['model'], as_index=False).mean()

# calcuale skill score
val_mean = df_mean_crps.query('model == "mean"').value.iat[0]
val_naive = df_mean_crps.query('model == "naive"').value.iat[0]
skill_score = (val_naive - val_mean) / val_naive
print(f"{val_mean=}")
print(f"{val_naive=}")
print(f"{skill_score=}")
```

```sh
> val_mean=0.033
> val_naive=0.023
> skill_score=-0.418
```

In general, the lower crps is the better. If skill score is negative that indicates that the naive model outperforms the alternative. Also, skill scores have the same form as "percent improvement" and my be interpreted the same way.