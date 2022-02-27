# Simple Example With Plotting

```python
import pandas as pd
import tablespoon as tbsp
from tablespoon.data import APPL


n = tbsp.Naive()
df_n = (n.predict(APPL, horizon=7*4, frequency="D", lag = 1, uncertainty_samples = 8000).assign(model = 'naive'))
print(df_n.head())

theme_set(theme_538)
palette = ["#000000", "#ee1d52"]

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
```