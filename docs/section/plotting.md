# Simple Example With Plotting

This is a very simple example using some data provided in the `tablespoon` package.

```python
import tablespoon as tbsp
from tablespoon.data import SEAS
from plotnine import *
import pandas as pd
from mizani.breaks import date_breaks

sn = tbsp.Snaive()
df_sn = sn.predict(
    SEAS, horizon=7 * 4, frequency="D", lag=7, uncertainty_samples=800
).assign(model="snaive")

df_sn.head(10)

theme_set(theme_538)
palette = ["#000000", "#ee1d52"]
df_SEAS = SEAS
df_SEAS = df_SEAS.assign(ds = lambda df: pd.to_datetime(df.ds))
df_actuals_forecasts_n = pd.concat([df_SEAS, df_sn])
p = (
    ggplot(df_actuals_forecasts_n, aes(x="ds", y="y"))
    + geom_line(aes(y = 'y'), color = palette[0])
    + geom_point(aes(y = 'y_sim'), color = palette[1], size = 0.1, alpha = 0.1)
    + scale_x_datetime(breaks=date_breaks("1 week"))
    + theme(axis_text_x=element_text(angle=45))
    + xlab("")
    + ggtitle("Snaive (Uncertainty expands on each seasonal horizon)")
    + scale_color_manual(palette)
)
p.save(filename="forecasts_n.jpg", width=14, height=3)
```
