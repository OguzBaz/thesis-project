// Hybrid Reinforcement Learning Model (two-step task)
// Corrected version with proper stickiness handling

data {
  int<lower=1> S;
  array[S] int<lower=0> T;
  int<lower=1> T_max;

  array[S, T_max] int<lower=1,upper=2> c1;
  array[S, T_max] int<lower=1,upper=2> c2;
  array[S, T_max] real r;
  array[S, T_max] int<lower=1,upper=3> s2raw;
  array[S] int<lower=1,upper=2> prior_choice;
  real<lower=0,upper=1> t_common;
}

parameters {
  vector<lower=1e-6, upper=1-1e-6>[S] alpha;
  vector<lower=1e-6, upper=1-1e-6>[S] lambda_;
  vector<lower=1e-6>[S] beta_mb;
  vector<lower=1e-6>[S] beta_mf;
  vector<lower=1e-6>[S] beta2;
  vector[S] stickiness;
}

model {
  alpha      ~ beta(1.1, 1.1);
  lambda_    ~ beta(1.1, 1.1);
  beta_mb    ~ gamma(3, 1);
  beta_mf    ~ gamma(3, 1);
  beta2      ~ gamma(3, 1);
  stickiness ~ normal(0, 10);

  for (s in 1:S) {
    matrix[3,2] Q = rep_matrix(0, 3, 2);
    int prev = prior_choice[s];

    for (t in 1:T[s]) {
      int s2 = (s2raw[s,t] >= 2) ? s2raw[s,t] : 2;

      real maxA = fmax(Q[2,1], Q[2,2]);
      real maxB = fmax(Q[3,1], Q[3,2]);

      real Qmb1 = t_common      * maxA + (1 - t_common) * maxB;
      real Qmb2 = (1 - t_common)* maxA + t_common       * maxB;

      real Qmf1 = Q[1,1];
      real Qmf2 = Q[1,2];

      // *** FIXED stickiness ***
      real rep_bias = (prev == 1 ? 1 : (prev == 2 ? -1 : 0)) * stickiness[s];

      real logit_s1 = beta_mf[s] * (Qmf1 - Qmf2)
                    + beta_mb[s] * (Qmb1 - Qmb2)
                    + rep_bias;

      target += bernoulli_logit_lpmf(c1[s,t] == 1 | logit_s1);

      real logit_s2 = beta2[s] * (Q[s2,1] - Q[s2,2]);

      target += bernoulli_logit_lpmf(c2[s,t] == 1 | logit_s2);

      real delta_rew = r[s,t] - Q[s2, c2[s,t]];
      Q[s2, c2[s,t]] += alpha[s] * delta_rew;

      real delta_state = Q[s2, c2[s,t]] - Q[1, c1[s,t]];
      Q[1, c1[s,t]] += alpha[s] * delta_state
                       + lambda_[s] * alpha[s] * delta_rew;

      prev = c1[s,t];
    }
  }
}

generated quantities {
  vector[S] beta1_stage1 = beta_mb + beta_mf;
  vector[S] w_hybrid;
  array[S, T_max] real log_lik;
  array[S, T_max] int y1_rep;
  array[S, T_max] int y2_rep;

  for (s in 1:S) {
    matrix[3,2] Q = rep_matrix(0, 3, 2);
    int prev = prior_choice[s];

    for (t in 1:T[s]) {
      int s2 = (s2raw[s,t] >= 2) ? s2raw[s,t] : 2;

      real maxA = fmax(Q[2,1], Q[2,2]);
      real maxB = fmax(Q[3,1], Q[3,2]);

      real Qmb1 = t_common      * maxA + (1 - t_common) * maxB;
      real Qmb2 = (1 - t_common)* maxA + t_common       * maxB;

      real Qmf1 = Q[1,1];
      real Qmf2 = Q[1,2];

      real rep_bias = (prev == 1 ? 1 : (prev == 2 ? -1 : 0)) * stickiness[s];

      real logit_s1 = beta_mf[s] * (Qmf1 - Qmf2)
                    + beta_mb[s] * (Qmb1 - Qmb2)
                    + rep_bias;

      y1_rep[s,t] = bernoulli_logit_rng(logit_s1) ? 1 : 2;
      real logit_s2 = beta2[s] * (Q[s2,1] - Q[s2,2]);
      y2_rep[s,t] = bernoulli_logit_rng(logit_s2) ? 1 : 2;

      log_lik[s,t] = bernoulli_logit_lpmf(c1[s,t] == 1 | logit_s1)
                   + bernoulli_logit_lpmf(c2[s,t] == 1 | logit_s2);

      real delta_rew = r[s,t] - Q[s2, c2[s,t]];
      Q[s2, c2[s,t]] += alpha[s] * delta_rew;

      real delta_state = Q[s2, c2[s,t]] - Q[1, c1[s,t]];
      Q[1, c1[s,t]] += alpha[s] * delta_state
                       + lambda_[s] * alpha[s] * delta_rew;

      prev = c1[s,t];
    }

    for (t in (T[s]+1):T_max) {
      log_lik[s,t] = 0;
      y1_rep[s,t] = 1;
      y2_rep[s,t] = 1;
    }

    w_hybrid[s] = beta_mb[s] / (beta_mb[s] + beta_mf[s]);
  }
}
