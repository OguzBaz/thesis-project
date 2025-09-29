// Stan Hybrid Reinforcement Learning Model (two-step task, MB/MF hybrid)
// Eqs: (1) TD learning with eligibility lambda
//      (2) MB expected-max with t_common
//      (3) Hybrid softmax (S1) + softmax (S2)
// Priors: alpha, lambda ~ Beta(1.1,1.1); betas ~ Gamma(3,1); stickiness ~ Normal(0,10)

data {
  int<lower=1> S;
  int<lower=1> T_max;
  array[S] int<lower=0> T;
  array[S, T_max] int<lower=1,upper=2> c1;
  array[S, T_max] int<lower=1,upper=2> c2;
  array[S, T_max] real r;
  array[S, T_max] int<lower=1,upper=3> s2raw;
  array[S] int<lower=1,upper=2> prior_choice;
  real<lower=0,upper=1> t_common;
}

parameters {
  // subject-level parameters (independent priors, as in MATLAB fits)
  vector<lower=1e-6, upper=1-1e-6>[S] alpha;     // learning rate
  vector<lower=1e-6, upper=1-1e-6>[S] lambda_;   // eligibility trace
  vector<lower=1e-6>[S] beta_mb;                 // S1 MB inverse-temp weight
  vector<lower=1e-6>[S] beta_mf;                 // S1 MF inverse-temp weight
  vector<lower=1e-6>[S] beta2;                   // S2 inverse temperature
  vector[S] stickiness;                          // perseveration bias (real)
}

model {
  // ----- Priors (MAP-style) -----
  alpha      ~ beta(1.1, 1.1);
  lambda_    ~ beta(1.1, 1.1);
  beta_mb    ~ gamma(3, 1);
  beta_mf    ~ gamma(3, 1);
  beta2      ~ gamma(3, 1);
  stickiness ~ normal(0, 10);

  // ----- Likelihood -----
  for (s in 1:S) {
    // Q table: row 1 = S1; rows 2 & 3 = second-stage states; each has 2 actions
    matrix[3,2] Q = rep_matrix(0, 3, 2);
    int prev = prior_choice[s]; // for S1 stickiness on first modeled trial

    for (t in 1:T[s]) {
      // Map to rows 2 or 3 (accept 2/3; if data uses 1/2, still works)
      int s2;
      if (s2raw[s,t] == 3) s2 = 3;
      else if (s2raw[s,t] == 2) s2 = 2;
      else s2 = s2raw[s,t];

      // (2) Model-based expected-max backup at stage-1:
      real maxA = fmax(Q[2,1], Q[2,2]); // best action if landing in state A (row 2)
      real maxB = fmax(Q[3,1], Q[3,2]); // best action if landing in state B (row 3)
      real Qmb1 = t_common      * maxA + (1 - t_common) * maxB; // pick S1 action 1
      real Qmb2 = (1 - t_common)* maxA + t_common       * maxB; // pick S1 action 2

      // MF cached S1 values
      real Qmf1 = Q[1,1];
      real Qmf2 = Q[1,2];

      // Stickiness bias: +p for repeating action 1, -p if previous was 2
      real rep_bias = (prev == 1 ? 1 : (prev == 2 ? -1 : 0)) * stickiness[s];

      // (3) Stage-1 hybrid softmax: Bernoulli-logit on (a1==1) with hybrid value diff
      real logit_s1 = beta_mf[s] * (Qmf1 - Qmf2)
                    + beta_mb[s] * (Qmb1 - Qmb2)
                    + rep_bias;
      target += bernoulli_logit_lpmf( c1[s,t] == 1 | logit_s1 );

      // Stage-2 softmax: Bernoulli-logit on (a2==1) with beta2 * Q-diff
      real logit_s2 = beta2[s] * (Q[s2,1] - Q[s2,2]);
      target += bernoulli_logit_lpmf( c2[s,t] == 1 | logit_s2 );

      // (1) TD learning with eligibility:
      // reward PE at stage-2
      real delta_rew = r[s,t] - Q[s2, c2[s,t]];
      Q[s2, c2[s,t]] += alpha[s] * delta_rew;

      // state-PE at stage-1 plus lambda back-prop of reward PE
      real delta_state = Q[s2, c2[s,t]] - Q[1, c1[s,t]];
      Q[1, c1[s,t]] += alpha[s] * delta_state + lambda_[s] * alpha[s] * delta_rew;

      // update prev choice for next trial's stickiness
      prev = c1[s,t];
    }
  }
}

generated quantities {
  // handy reparameterizations for reporting
  vector[S] beta1_stage1 = beta_mb + beta_mf;       // overall S1 inverse temperature
  vector[S] w_hybrid;
  for (s in 1:S) w_hybrid[s] = beta_mb[s] / (beta_mb[s] + beta_mf[s]);
}