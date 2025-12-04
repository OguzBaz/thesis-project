data {
  int<lower=1> S;
  array[S] int<lower=1> T;
  int<lower=1> T_max;

  array[S, T_max] int<lower=1, upper=2> c1;
  array[S, T_max] int<lower=1, upper=2> c2;
  array[S, T_max] real r;
  array[S, T_max] int<lower=2, upper=3> s2raw;
  array[S, T_max] real<lower=0> rt2;

  array[S] real rt2_min;
  real<lower=0> ter_eps;

  array[S] int<lower=1, upper=2> prior_choice;
  real<lower=0, upper=1> t_common;
}

parameters {
  vector[S] alpha_raw;
  vector[S] beta_mf_raw;
  vector[S] beta_mb_raw;
  vector[S] w_raw;
  vector[S] stick_raw;

  vector[S] a_raw;
  vector[S] Ter_raw;
  vector[S] vmod_raw;
  vector[S] z_raw;
}

transformed parameters {
  vector<lower=0,upper=1>[S] alpha;
  vector<lower=0,upper=10>[S] beta_mf;
  vector<lower=0,upper=10>[S] beta_mb;
  vector<lower=0,upper=1>[S] w;
  vector[S] stick;

  vector<lower=0.3, upper=3>[S] a;
  vector<lower=0.05, upper=1>[S] Ter;
  vector<lower=0.1, upper=5>[S] vmod;
  vector<lower=0.2, upper=0.8>[S] z;

  for (s in 1:S) {
    alpha[s]    = inv_logit(alpha_raw[s]);
    beta_mf[s]  = 10 * inv_logit(beta_mf_raw[s]);
    beta_mb[s]  = 10 * inv_logit(beta_mb_raw[s]);
    w[s]        = inv_logit(w_raw[s]);
    stick[s]    = stick_raw[s] * 0.5;

    a[s] = 0.3 + 2.7 * inv_logit(a_raw[s]);
    vmod[s] = 0.1 + 4.9 * inv_logit(vmod_raw[s]);

    real upperTer = rt2_min[s] - ter_eps;
    if (upperTer < 0.06)
      upperTer = 0.06;

    Ter[s] = 0.05 + (upperTer - 0.05) * inv_logit(Ter_raw[s]);

    z[s] = 0.2 + 0.6 * inv_logit(z_raw[s]);
  }
}

model {

  alpha_raw ~ normal(0, 1);
  beta_mf_raw ~ normal(0, 1);
  beta_mb_raw ~ normal(0, 1);
  w_raw ~ normal(0, 1);
  stick_raw ~ normal(0, 1);

  a_raw ~ normal(0, 1);
  Ter_raw ~ normal(0, 1);
  vmod_raw ~ normal(0, 1);
  z_raw ~ normal(0, 1);

  for (s in 1:S) {

    vector[2] Qs;
    Qs[1] = 0;
    Qs[2] = 0;

    for (t in 1:T[s]) {

      int st = s2raw[s,t] - 1;

      real Qmf = Qs[st];
      real Qmb = 0.5 * (Qs[1] + Qs[2]);
      real Qhyb = w[s] * Qmb + (1 - w[s]) * Qmf;

      real util1 = beta_mf[s] * Qhyb + stick[s] * (prior_choice[s] == c1[s,t]);

      target += bernoulli_logit_lpmf(c1[s,t] - 1 | util1);

      // SAFE DRIFT
      real drift = vmod[s] * (beta_mf[s] * Qmf);
      real drift_safe = drift;

      if (drift_safe < 0.01) drift_safe = 0.01;
      if (drift_safe > 5)    drift_safe = 5;

      target += wiener_lpdf(rt2[s,t] |
                            a[s],
                            Ter[s],
                            z[s],
                            drift_safe);

      Qs[st] = Qs[st] + alpha[s] * (r[s,t] - Qs[st]);
    }
  }
}

generated quantities {
  array[S, T_max] int y1_rep;
  array[S, T_max] real y2_rep;
  array[S, T_max] real log_lik;

  for (s in 1:S) {

    vector[2] Qs;
    Qs[1] = 0;
    Qs[2] = 0;

    for (t in 1:T[s]) {

      int st = s2raw[s,t] - 1;

      real Qmf = Qs[st];
      real Qmb = 0.5 * (Qs[1] + Qs[2]);
      real Qhyb = w[s] * Qmb + (1 - w[s]) * Qmf;

      real util1 = beta_mf[s] * Qhyb + stick[s] * (prior_choice[s] == c1[s,t]);

      // log likelihood
      log_lik[s,t] = bernoulli_logit_lpmf(c1[s,t] - 1 | util1);

      // replicate choice
      y1_rep[s,t] = bernoulli_logit_rng(util1) + 1;

      // safe drift again
      real drift = vmod[s] * (beta_mf[s] * Qmf);
      real drift_safe = drift;

      if (drift_safe < 0.01) drift_safe = 0.01;
      if (drift_safe > 5)    drift_safe = 5;

      // simple RT generator (not wiener_rng)
      real mean_rt = Ter[s] + (a[s] * z[s]) / drift_safe;
      real sd_rt   = 0.05 + fabs(0.2 / drift_safe);

      if (mean_rt < 0.05) mean_rt = 0.05;

      y2_rep[s,t] = normal_rng(mean_rt, sd_rt);

      if (y2_rep[s,t] < 0.05)
        y2_rep[s,t] = 0.05;

      Qs[st] = Qs[st] + alpha[s] * (r[s,t] - Qs[st]);
    }

    // pad trials if needed
    for (t in T[s] + 1:T_max) {
      y1_rep[s,t] = 1;
      y2_rep[s,t] = 0.1;
      log_lik[s,t] = 0;
    }
  }
}
