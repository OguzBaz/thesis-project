// Two-step Hybrid RL + Stage-2 DDM (RLDDM)
// - Stage 1: hybrid MB/MF softmax (+ stickiness)
// - Stage 2: Wiener DDM with drift v_t = vmod * (Q[s2,1] - Q[s2,2])
// - Learning: TD with eligibility lambda
//
// Notes:
// * RTs (rt2) must be in seconds.
// * To condition on the observed boundary, we keep z fixed and flip the drift sign if c2==2.

data {
  int<lower=1> S;                        // subjects
  int<lower=1> T_max;                    // max trials across subjects
  array[S] int<lower=0> T;               // trials used per subject (first T[s] rows)

  // per-subject, per-trial (only first T[s] are used)
  array[S, T_max] int<lower=1,upper=2> c1;      // stage-1 choice (1/2)
  array[S, T_max] int<lower=1,upper=2> c2;      // stage-2 choice (1/2) -> boundary
  array[S, T_max] real r;                        // reward outcome
  array[S, T_max] int<lower=1,upper=3> s2raw;   // second-stage state code (often 2/3)
  array[S, T_max] real<lower=0> rt2;            // stage-2 response time (seconds)

  array[S] int<lower=1,upper=2> prior_choice;   // S1 choice for first modeled trial
  real<lower=0,upper=1> t_common;               // common transition prob (e.g., 0.7)
}

parameters {
  // RL parameters
  vector<lower=1e-6, upper=1-1e-6>[S] alpha;     // learning rate
  vector<lower=1e-6, upper=1-1e-6>[S] lambda_;   // eligibility trace
  vector<lower=1e-6>[S] beta_mb;                 // S1 MB weight
  vector<lower=1e-6>[S] beta_mf;                 // S1 MF weight
  vector[S] stickiness;                          // S1 perseveration bias

  // DDM (stage-2) parameters
  vector<lower=1e-6>[S] vmod;                    // drift scaling
  vector<lower=1e-6>[S] a;                       // boundary separation
  vector<lower=1e-6>[S] Ter;                     // non-decision time
  vector<lower=1e-3, upper=1-1e-3>[S] z;         // starting-point fraction (0..1)
}

model {
  // --- Priors (tune as needed) ---
  alpha      ~ beta(1.1, 1.1);
  lambda_    ~ beta(1.1, 1.1);
  beta_mb    ~ gamma(3, 1);
  beta_mf    ~ gamma(3, 1);
  stickiness ~ normal(0, 10);

  vmod ~ gamma(3, 1);         // positive scale on value difference
  a    ~ lognormal(0, 0.5);   // > 0
  Ter  ~ lognormal(-1, 0.5);  // > 0  (seconds)
  z    ~ beta(1.5, 1.5);      // centered near 0.5

  // --- Likelihood ---
  for (s in 1:S) {
    // Q table: row 1 = S1; rows 2..3 = second-stage states; each has 2 actions
    matrix[3,2] Q = rep_matrix(0, 3, 2);
    int prev = prior_choice[s];

    for (t in 1:T[s]) {
      // map second-stage state: if dataset uses {1,2}, treat 1->2 and 2->3
      int s2 = (s2raw[s,t] >= 2) ? s2raw[s,t] : 2;  // {2,3} else map 1→2

      // ----- Stage 1 softmax (hybrid MB/MF + stickiness) -----
      // MB expected-max backup
      real maxA = fmax(Q[2,1], Q[2,2]);
      real maxB = fmax(Q[3,1], Q[3,2]);
      real Qmb1 = t_common      * maxA + (1 - t_common) * maxB; // action 1 at S1
      real Qmb2 = (1 - t_common)* maxA + t_common       * maxB; // action 2 at S1
      // MF cached S1 values
      real Qmf1 = Q[1,1];
      real Qmf2 = Q[1,2];
      // stickiness (+p if repeat action 1, -p if repeat action 2)
      real rep_bias = (prev == 1 ? 1 : (prev == 2 ? -1 : 0)) * stickiness[s];

      // Bernoulli-logit on a1==1
      real logit_s1 = beta_mf[s] * (Qmf1 - Qmf2)
                    + beta_mb[s] * (Qmb1 - Qmb2)
                    + rep_bias;
      target += bernoulli_logit_lpmf(c1[s,t] == 1 | logit_s1);

      // ----- Stage 2 DDM (Wiener) -----
      // trial-wise drift from value difference (before updating)
      real delta_Q = Q[s2,1] - Q[s2,2];
      real v_raw   = vmod[s] * delta_Q;   // this is v_t

      // condition on observed boundary by flipping drift only
      real drift_eff = (c2[s,t] == 1) ?  v_raw : -v_raw;
      target += wiener_lpdf(rt2[s,t] | a[s], Ter[s], z[s], drift_eff);

      // ----- TD(λ) updates -----
      real delta_rew   = r[s,t] - Q[s2, c2[s,t]];
      Q[s2, c2[s,t]]  += alpha[s] * delta_rew;

      real delta_state = Q[s2, c2[s,t]] - Q[1, c1[s,t]];
      Q[1, c1[s,t]]   += alpha[s] * delta_state + lambda_[s] * alpha[s] * delta_rew;

      prev = c1[s,t];
    }
  }
}

generated quantities {
  // Report hybrid stage-1 temp & weight (handy)
  vector[S] beta1_stage1 = beta_mb + beta_mf;
  vector[S] w_hybrid;
  for (s in 1:S) w_hybrid[s] = beta_mb[s] / (beta_mb[s] + beta_mf[s]);

  // Optional: per-trial v_t (not produced by `optimize()` runs)
  array[S, T_max] real v_t;
  for (s in 1:S) {
    matrix[3,2] Q = rep_matrix(0, 3, 2);
    int prev = prior_choice[s];
    for (t in 1:T[s]) {
      int s2 = (s2raw[s,t] >= 2) ? s2raw[s,t] : 2;

      real delta_Q = Q[s2,1] - Q[s2,2];
      v_t[s,t] = vmod[s] * delta_Q;  // save drift before learning update

      real delta_rew   = r[s,t] - Q[s2, c2[s,t]];
      Q[s2, c2[s,t]]  += alpha[s] * delta_rew;
      real delta_state = Q[s2, c2[s,t]] - Q[1, c1[s,t]];
      Q[1, c1[s,t]]   += alpha[s] * delta_state + lambda_[s] * alpha[s] * delta_rew;
      prev = c1[s,t];
    }
    for (t in (T[s]+1):T_max) v_t[s,t] = 0; // fill padding, ignored
  }
}
