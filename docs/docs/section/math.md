
# Mean

## Model

$$
y \sim \textsf{normal}\left(\bar{y}, \sigma\right)
$$

    y ~ normal(mean(y), sigma);

## Data Generation
$$
\textsf{forecast} \sim \textsf{student_t_rng}\left(T-1, \bar{y}, \sigma \sqrt{1 + \frac{1}{T}}\right)
$$

    forecast[h] = student_t_rng(T - 1, mean(y), sigma * sqrt(1 + (1 / T)));

# Naive

## Model

$$
y_t \sim \textsf{normal}\left(y_{t-1}, \sigma\right)
$$

    y[t] ~ normal(y[t-lag], sigma);

## Data Generation

$$
\textsf{forecast}_{h} \sim \textsf{normal_rng}\left(y_t, \sigma\sqrt{h}\right)
$$

# Seasonal Naive

## Model

$$
y_t \sim \textsf{normal}\left(y_{t-1}, \sigma\right)
$$

    y[t] ~ normal(y[t-lag], sigma);

## Data Generation

$$
\textsf{forecast}_{h} \sim \textsf{normal_rng}\left(y_{t - (lag - (h \textsf{ mod } lag))}, \sigma\sqrt{trunc(((h - 1) * 1) / (lag)) + 1)}\right)
$$

    forecast[h] = normal_rng(y[T - (lag - (h % lag))], sigma * sqrt(trunc(((h - 1) * 1) %/% (lag)) + 1));
