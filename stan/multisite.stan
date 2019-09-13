data {
  int<lower=1> N; // number of observations
  vector[N] se; // known standard errors
  vector[N] x; // observations
  real maxError;
  real mindf;
}
parameters {
  real mu;
  real<lower=0> sigma; // t scale
  real<lower=1> df;
  vector[N] T; // true site difference from mean
}
model {
  //priors
  df - mindf ~ exponential(.2);
  sigma - (maxError/2) ~ exponential(.2);
  mu ~ normal(0,20);

  //model
  T ~ student_t(df, mu, sigma);
  x ~ normal(T + mu, se);
}
