data {
  int<lower=1> N; // number of observations
  vector[N] se; // known standard errors
  vector[N] x; // observations
  real maxError;
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
  T ~ student_t(exp(d) + mindf, 0., exp(varsigma) + maxError/2);
  x ~ normal(T + mu, se);
}
