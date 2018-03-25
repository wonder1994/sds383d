// saved as 8schools.stan
data {
  int<lower=0> J; // number of schools 
  int<lower=5> y[J]; // 
  real x[J]; // 
}
parameters {
  real beta[2]; 
}
transformed parameters {
  real lambda[J];
  for (j in 1:J){
    lambda[j] = exp(beta[1] + x[j]* beta[2]);
  }
}
model {
  beta[1] ~ normal(0,1);
  beta[2] ~ normal(0,1);
  y ~ poisson(lambda);
}
