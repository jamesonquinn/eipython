data {
  int<lower=1> N; // number of observations
  vector[N] se; // known standard errors
  vector[N] x; // observations
  real maxError;
  real MIN_SIGMA_OVER_S;
  real mindf;
  real dmean;
  real smean;
  real dscale;
  real sscale;
}
parameters {
  real mu;
  real varsigma; // t scale
  real d;
  vector[N] T; // true site difference from mean
}
model {
  //priors
  d ~ normal(dmean,dscale);
  varsigma ~ normal(smean,sscale);
  mu ~ normal(0,20.0);

  //model
  T/(exp(varsigma) + maxError*MIN_SIGMA_OVER_S) ~ student_t(exp(d) + mindf, 0., 1.);
  x ~ normal(T + mu, se);
}
