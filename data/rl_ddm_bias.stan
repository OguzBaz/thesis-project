functions {
  // Custom signed Wiener function: maps choice 2 (lower boundary) to negative RT.
  real wiener_signed_lpdf(real Y, real boundary, real ndt, real bias, real drift) {
    if (Y >= 0) {
      // Choice 1 (Positive RT, Upper Boundary)
      return wiener_lpdf(abs(Y) | boundary, ndt, bias, drift);
    } else {
      // Choice 2 (Negative RT, Lower Boundary)
      return wiener_lpdf(abs(Y) | boundary, ndt, 1 - bias, -drift);
    }
  }
}

data {
  int<lower=1> S;
  int<lower=1> T_max;
  array[S] int<lower=0> T;

  // Data allowing padding (0) for missed trials
  array[S, T_max] int<lower=0, upper=2> c1;
  array[S, T_max] int<lower=0, upper=3> s2;
  array[S, T_max] int<lower=0, upper=2> c2;
  array[S, T_max] real r;
  array[S, T_max] real rt2; 
  real<lower=0,upper=1> t_common; // Transition probability
}

transformed data {
  array[S, T_max] real rt_signed;
  real min_rt = 0.001; 

  for (s in 1:S) {
    for (t in 1:T_max) {
      if (t <= T[s] && c2[s,t] != 0) {
        real rt_seconds = rt2[s,t] / 1000.0;
        rt_seconds = fmax(rt_seconds, min_rt); // Ensure RT > min_rt (for safety)

        if (c2[s,t] == 1) {
           rt_signed[s,t] = rt_seconds;
        } else {
           rt_signed[s,t] = -rt_seconds;
        }
      } else {
        rt_signed[s,t] = 0.0;
      }
    }
  }
}

parameters {
  // --- RL Parameters ---
  vector<lower=1e-6, upper=1-1e-6>[S] alpha;     
  vector<lower=1e-6, upper=1-1e-6>[S] lambda;    
  vector<lower=1e-6, upper=1-1e-6>[S] omega;     
  vector<lower=1e-6>[S] beta_s1;                 
  vector[S] pers;                                

  // --- DDM Parameters ---
  vector<lower=0>[S] v_coeff; 
  vector<lower=0>[S] a;       
  vector<lower=0>[S] ter;     
  vector<lower=0, upper=1>[S] z;       
}

model {
  // --- RL Priors (Proven Effective) ---
  alpha   ~ beta(1.1, 1.1);
  lambda  ~ beta(1.1, 1.1);
  omega   ~ beta(1.1, 1.1);
  beta_s1 ~ gamma(3, 1);
  pers    ~ normal(0, 10);

  // --- DDM Priors (Tighter for Stability, prevents Ter crash) ---
  
  // Non-Decision Time (ter > 0): Mean 0.1s, tight SD 0.01s (to avoid RT > Ter crash)
  ter ~ normal(0.1, 0.01) T[0, ]; 

  // Boundary Separation (a > 0): Mean 0.7s (Tighter start)
  a ~ normal(0.7, 0.2) T[0, ];

  // Starting Point Bias (0 < z < 1)
  z ~ beta(1.1, 1.1);

  // Drift Coefficient Scaling (v_coeff > 0)
  v_coeff ~ gamma(1, 1);
  
  // --- LIKELIHOOD ---
  for (s in 1:S) {
    matrix[3,2] Q = rep_matrix(0.5, 3, 2);
    int prev_c1 = 0;

    for (t in 1:T[s]) {
      if (c2[s,t] == 0) { 
        prev_c1 = 0;
        continue;
      }
      
      // RL Value Calculations 
      real max_s2 = fmax(Q[2,1], Q[2,2]);
      real max_s3 = fmax(Q[3,1], Q[3,2]);
      real q_mb_1 = t_common * max_s2 + (1 - t_common) * max_s3;
      real q_mb_2 = (1 - t_common) * max_s2 + t_common * max_s3;
      real q_mf_1 = Q[1,1];
      real q_mf_2 = Q[1,2];
      real q_net_1 = omega[s] * q_mb_1 + (1 - omega[s]) * q_mf_1;
      real q_net_2 = omega[s] * q_mb_2 + (1 - omega[s]) * q_mf_2;
      
      real stick = 0.0;
      if (prev_c1 == 1) stick = pers[s];
      else if (prev_c1 == 2) stick = -pers[s];

      // --- STAGE 1 LIKELIHOOD (Bernoulli) ---
      if (c1[s,t] != 0) {
        // *** CRITICAL FIX: Reverse Q-difference sign (Q2 - Q1) ***
        real logit_s1 = beta_s1[s] * (q_net_2 - q_net_1) + stick; 
        
        target += bernoulli_logit_lpmf(c1[s,t] - 1 | logit_s1);
      }

      // --- STAGE 2 LIKELIHOOD (DDM/Wiener) ---
      int state_idx = s2[s,t];
      real delta_Q = Q[state_idx, 1] - Q[state_idx, 2];
      real drift_t = v_coeff[s] * delta_Q; 
      
      target += wiener_signed_lpdf(rt_signed[s,t] | a[s], ter[s], z[s], drift_t);
      
      // --- Q-VALUE UPDATES ---
      int act2 = c2[s,t];
      real pe2 = r[s,t] - Q[state_idx, act2];
      Q[state_idx, act2] += alpha[s] * pe2;

      if (c1[s,t] != 0) {
        int act1 = c1[s,t];
        real val_next = Q[state_idx, act2];
        real pe1 = val_next - Q[1, act1];
        Q[1, act1] += alpha[s] * pe1 + lambda[s] * alpha[s] * pe2;
        prev_c1 = act1;
      } else {
        prev_c1 = 0;
      }
    }
  }
}

generated quantities {
  array[S, T_max] real log_lik_trial;
  array[S, T_max] real accuracy_s1;
  array[S, T_max] real accuracy_s2;
  array[S, T_max] real v_t;

  for (s in 1:S) {
    matrix[3,2] Q = rep_matrix(0.5, 3, 2);
    int prev_c1 = 0;

    for (t in 1:T_max) {
      if (t > T[s] || c2[s,t] == 0) {
          log_lik_trial[s,t] = 0;
          accuracy_s1[s,t] = 0;
          accuracy_s2[s,t] = 0;
          v_t[s,t] = 0;
          continue;
      }
      
      // RL Value Calculations (REPLICATION)
      real max_s2 = fmax(Q[2,1], Q[2,2]);
      real max_s3 = fmax(Q[3,1], Q[3,2]);
      real q_mb_1 = t_common * max_s2 + (1 - t_common) * max_s3;
      real q_mb_2 = (1 - t_common) * max_s2 + t_common * max_s3;
      real q_mf_1 = Q[1,1];
      real q_mf_2 = Q[1,2];
      real q_net_1 = omega[s] * q_mb_1 + (1 - omega[s]) * q_mf_1;
      real q_net_2 = omega[s] * q_mb_2 + (1 - omega[s]) * q_mf_2;
      
      real stick = 0.0;
      if (prev_c1 == 1) stick = pers[s];
      else if (prev_c1 == 2) stick = -pers[s];

      // Stage 1 Logit & Prediction
      real logit_s1 = beta_s1[s] * (q_net_2 - q_net_1) + stick; 

      if (c1[s,t] != 0) {
        log_lik_trial[s,t] = bernoulli_logit_lpmf(c1[s,t] - 1 | logit_s1);
        int pred_s1 = (logit_s1 >= 0) ? 2 : 1;
        accuracy_s1[s,t] = (pred_s1 == c1[s,t]) ? 1.0 : 0.0;
      } else {
        log_lik_trial[s,t] = 0;
        accuracy_s1[s,t] = 0.0; 
      }

      // Stage 2 DDM & Prediction
      int state_idx = s2[s,t];
      real delta_Q = Q[state_idx, 1] - Q[state_idx, 2];
      v_t[s,t] = v_coeff[s] * delta_Q;
      
      // DDM Log-Likelihood 
      log_lik_trial[s,t] += wiener_signed_lpdf(rt_signed[s,t] | a[s], ter[s], z[s], v_t[s,t]);
      
      // DDM Choice Prediction 
      int pred_s2 = (v_t[s,t] >= 0) ? 1 : 2;
      accuracy_s2[s,t] = (pred_s2 == c2[s,t]) ? 1.0 : 0.0;

      // Q-VALUE UPDATES (REPLICATION)
      int act2 = c2[s,t];
      real pe2 = r[s,t] - Q[state_idx, act2];
      Q[state_idx, act2] += alpha[s] * pe2;

      if (c1[s,t] != 0) {
        int act1 = c1[s,t];
        real val_next = Q[state_idx, act2];
        real pe1 = val_next - Q[1, act1];
        Q[1, act1] += alpha[s] * pe1 + lambda[s] * alpha[s] * pe2;
        prev_c1 = act1;
      } else {
        prev_c1 = 0;
      }
    }
  }
}