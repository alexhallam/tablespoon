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
    # student_t_rng(nu, mu, sigma)
    forecast[h] = student_t_rng(T - 1, mean(y), sigma * sqrt(1 + (1 / T)));
  }
}
