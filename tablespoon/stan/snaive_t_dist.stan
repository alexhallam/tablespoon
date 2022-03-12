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
    # `%` is modulus and `%/%` is integer division
    # student_t_rng(nu, mu, sigma)
    forecast[h] = student_t_rng(T-1, y[T - (lag - (h % lag))], sigma * sqrt(trunc(((h - 1) * 1) / (lag)) + 1));
  }
}