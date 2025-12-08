// Hybrid Reinforcement Learning Model (two-step task)
// Using single beta (beta1) and a hybrid weight (w_mb)
data {
  int<lower=1> S;
  array[S] int<lower=0> T;
  int<lower=1> T_max;

  // FIX: Change <lower=1> to <lower=0> or remove the constraint entirely
  array[S, T_max] int<lower=0,upper=2> c1; // Allow 0 for padding
  array[S, T_max] int<lower=0,upper=2> c2; // Allow 0 for padding
  
  array[S, T_max] real r;
  array[S, T_max] int<lower=0,upper=3> s2raw; 
  array[S] int<lower=1,upper=2> prior_choice;
  real<lower=0,upper=1> t_common;
}

parameters {
  // RL Parameters
  vector<lower=1e-6, upper=1-1e-6>[S] alpha;      // Learning rate
  vector<lower=1e-6, upper=1-1e-6>[S] lambda_;    // Eligibility trace
  
  // Inverse Temperatures and Weight
  vector<lower=1e-6>[S] beta1;                    // Total Stage 1 Beta (Beta_MF + Beta_MB)
  vector<lower=1e-6>[S] beta2;                    // Stage 2 Beta
  vector<lower=1e-6, upper=1-1e-6>[S] w_mb;       // MB Weight (w)
  
  // Stickiness
  vector[S] stickiness;
}

model {
  // Priors
  alpha    ~ beta(1.1, 1.1);
  lambda_  ~ beta(1.1, 1.1);
  w_mb     ~ beta(1.1, 1.1);
  beta1    ~ gamma(3, 1);
  beta2    ~ gamma(3, 1);
  stickiness ~ normal(0, 10);

  for (s in 1:S) {
    matrix[3,2] Q = rep_matrix(0, 3, 2);
    int prev = prior_choice[s];

    for (t in 1:T[s]) {
      int s2 = (s2raw[s,t] >= 2) ? s2raw[s,t] : 2;

      // MB Q-Values
      real maxA = fmax(Q[2,1], Q[2,2]);
      real maxB = fmax(Q[3,1], Q[3,2]);
      real Qmb1 = t_common    * maxA + (1 - t_common) * maxB;
      real Qmb2 = (1 - t_common)* maxA + t_common    * maxB;

      // MF Q-Values
      real Qmf1 = Q[1,1];
      real Qmf2 = Q[1,2];

      // Choice Bias (Stickiness)
      real rep_bias = (prev == 1 ? 1 : (prev == 2 ? -1 : 0)) * stickiness[s];

      // Stage 1 Logit (Hybrid Decision Rule)
      real delta_mf = Qmf1 - Qmf2;
      real delta_mb = Qmb1 - Qmb2;
      
      real logit_s1 = beta1[s] * ((1 - w_mb[s]) * delta_mf + w_mb[s] * delta_mb)
                        + rep_bias;

      target += bernoulli_logit_lpmf(c1[s,t] == 1 | logit_s1);

      // Stage 2 Logit
      real logit_s2 = beta2[s] * (Q[s2,1] - Q[s2,2]);
      target += bernoulli_logit_lpmf(c2[s,t] == 1 | logit_s2);

      // Q-Value Updates (Model-Free)
      // 1. Final outcome to State 2 Q-values
      real delta_rew = r[s,t] - Q[s2, c2[s,t]];
      Q[s2, c2[s,t]] += alpha[s] * delta_rew;

      // 2. State 2 Q-values to State 1 Q-values (plus eligibility trace)
      real delta_state = Q[s2, c2[s,t]] - Q[1, c1[s,t]];
      Q[1, c1[s,t]] += alpha[s] * delta_state
                         + lambda_[s] * alpha[s] * delta_rew; // SARSA(lambda) term
      prev = c1[s,t];
    }
  }
}

generated quantities {
  array[S, T_max] real log_lik;
  array[S, T_max] int y1_rep;
  array[S, T_max] int y2_rep;

  for (s in 1:S) {
    matrix[3,2] Q = rep_matrix(0, 3, 2);
    int prev = prior_choice[s];

    for (t in 1:T[s]) {
      int s2 = (s2raw[s,t] >= 2) ? s2raw[s,t] : 2;

      // MB Q-Values
      real maxA = fmax(Q[2,1], Q[2,2]);
      real maxB = fmax(Q[3,1], Q[3,2]);
      real Qmb1 = t_common    * maxA + (1 - t_common) * maxB;
      real Qmb2 = (1 - t_common)* maxA + t_common    * maxB;

      // MF Q-Values
      real Qmf1 = Q[1,1];
      real Qmf2 = Q[1,2];

      // Choice Bias
      real rep_bias = (prev == 1 ? 1 : (prev == 2 ? -1 : 0)) * stickiness[s];

      // Stage 1 Logit (Hybrid Decision Rule)
      real delta_mf = Qmf1 - Qmf2;
      real delta_mb = Qmb1 - Qmb2;
      
      real logit_s1 = beta1[s] * ((1 - w_mb[s]) * delta_mf + w_mb[s] * delta_mb)
                        + rep_bias;

      // Stage 2 Logit
      real logit_s2 = beta2[s] * (Q[s2,1] - Q[s2,2]);

      // Predictive Choices
      y1_rep[s,t] = bernoulli_logit_rng(logit_s1) ? 1 : 2;
      y2_rep[s,t] = bernoulli_logit_rng(logit_s2) ? 1 : 2;

      // Log-Likelihood
      log_lik[s,t] = bernoulli_logit_lpmf(c1[s,t] == 1 | logit_s1)
                   + bernoulli_logit_lpmf(c2[s,t] == 1 | logit_s2);

      // Q-Value Updates (same as model block)
      real delta_rew = r[s,t] - Q[s2, c2[s,t]];
      Q[s2, c2[s,t]] += alpha[s] * delta_rew;

      real delta_state = Q[s2, c2[s,t]] - Q[1, c1[s,t]];
      Q[1, c1[s,t]] += alpha[s] * delta_state
                         + lambda_[s] * alpha[s] * delta_rew;

      prev = c1[s,t];
    }
    
    // Fill remaining T_max slots with dummy data/zero log-lik
    for (t in (T[s]+1):T_max) {
      log_lik[s,t] = 0;
      y1_rep[s,t] = 1;
      y2_rep[s,t] = 1;
    }
  }
}