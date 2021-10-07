<h1 align="center">Forecast Methods</h1>

<p align="center"> tablespoon </p>

<p align="center">This page may seem redundant with code already present in the repository. That is okay! The goal is to be as clear as possible with how forecasts are being generated.</p>

# Forecast Methods

* [Mean](#mean)
* [Naive](#naive)
* [Seasonal Naive](#seasonal-naive)

# Mean

```stan
data {
  int<lower=0> T;
  int<lower=0> horizon;
  vector[T] y;
}

parameters {
  real<lower=0> sigma;
}

model {
  y ~ normal(mean(y), sigma);
}

generated quantities {
  vector[horizon] forecast;
  for (h in 1:horizon){
    forecast[h] = student_t_rng(T-1,mean(y),sigma * sqrt(1 + (1/T)));
  }
}
```

# Naive 

```stan
data {
  int<lower=0> T;
  int<lower=0> horizon;
  int<lower=0> lag;
  vector[T] y;
}

parameters {
  real<lower=0> sigma;
}

model {
  for(t in lag+1:T){
    y[t] ~ normal(y[t-lag], sigma);
  }
}

generated quantities {
  vector[horizon] forecast;
  for (h in 1:horizon){
    forecast[h] = normal_rng(y[T], sigma*sqrt(h));
  }
}
```


# Seasonal Naive

```stan
data {
  int<lower=0> T;
  int<lower=0> horizon;
  int<lower=0> lag;
  vector[T] y;
}

parameters {
  real<lower=0> sigma;
}

model {
for(t in lag+1:T)
  y[t] ~ normal(y[t-lag], sigma);
}

generated quantities {
  vector[horizon] forecast;
  for (h in 1:horizon){
    forecast[h] = normal_rng(y[T-(lag-(h%lag))], sigma*sqrt(trunc((h-1)/(lag)) + 1));
  }
}
```
