# RL, Model-Based RL, and RL-DDM Model Equations

## 1. Model-Free Reinforcement Learning (Q-Learning)

$$
Q_{t+1}(s, a) = Q_t(s, a) + \alpha \cdot \left[r_t - Q_t(s, a)\right]
$$

- $Q_t(s, a)$: Expected value for action $a$ in state $s$ at trial $t$  
- $\alpha$: Learning rate ($0 < \alpha < 1$)  
- $r_t$: Reward received at trial $t$

---

## 2. Model-Based Value Calculation (for Two-Step Task)

The model-based value at the first stage is computed by considering transition probabilities and second-stage Q-values:

$$
Q_\mathrm{MB}(s_1, a) = \sum_{s_2} P(s_2 \mid s_1, a) \cdot \max_{a'} Q(s_2, a')
$$

- $P(s_2 \mid s_1, a)$: Transition probability from state $s_1$ to $s_2$ given action $a$
- $\max_{a'} Q(s_2, a')$: Maximum Q-value at second-stage state $s_2$

---

## 3. Softmax Action Selection

$$
P(a_t = a) = \frac{\exp\left(\beta \cdot Q_t(s, a)\right)}{\sum_{a'} \exp\left(\beta \cdot Q_t(s, a')\right)}
$$

- $\beta$: Inverse temperature (decision noise parameter)

---

## 4. Hybrid Model-Based and Model-Free RL

$$
Q_\mathrm{hybrid}(s, a) = w \cdot Q_\mathrm{MB}(s, a) + (1 - w) \cdot Q_\mathrm{MF}(s, a)
$$

- $Q_\mathrm{MB}$: Model-based value (see above)  
- $Q_\mathrm{MF}$: Model-free value  
- $w$: Model-based weighting parameter ($0 \leq w \leq 1$)

---

## 5. Drift Diffusion Model (DDM) – Linking RL to Decision Process

### a. Drift Rate

$$
v_t = v \cdot \left[ Q_\mathrm{hybrid}(s, a_1) - Q_\mathrm{hybrid}(s, a_2) \right]
$$


- $v$: Drift scaling parameter  
- $a_1, a_2$: Available actions at trial $t$

### b. DDM Choice and RT Likelihood

$$
\text{RT}_t,\, a_t \sim \mathrm{DDM}(v_t, a, T_{er})
$$

- $a$: Boundary separation  
- $T_{er}$: Non-decision time  
- $v_t$: Drift rate for trial $t$

---

## 6. Parameter Table

| Parameter      | Description                                 |
| -------------- | ------------------------------------------- |
| $\alpha$       | Learning rate (0–1)                         |
| $\beta$        | Inverse temperature (softmax, positive)     |
| $w$            | Model-based RL weight (0–1)                 |
| $v$            | Drift scaling (positive)                    |
| $a$            | Boundary separation (positive)              |
| $T_{er}$       | Non-decision time (positive)                |

---




flowchart TD
    A[Unpack params & init Q-values] --> B[For each trial]
    B --> C[Update Q-values]
    C --> D[Compute hybrid Q]
    D --> E[Add stickiness]
    E --> F[Softmax & log-likelihood]
    F --> G[Accumulate loglik]
    G --> H[Return -loglik]


# Hybrid RL Model Fitting: Explanations

## hybrid_rl_negloglik

### What?

This function calculates the negative log-likelihood of a sequence of choices under a hybrid reinforcement learning (RL) model that combines model-free and model-based learning, with an added stickiness parameter to account for choice repetition.

### How?

- Unpacks model parameters and initializes Q-value arrays for both model-free and model-based learning.
- For each trial, updates Q-values using Q-learning and eligibility traces, computes model-based Q-values using a fixed transition structure, and combines them into a hybrid Q-value.
- Adds a stickiness term to the hybrid Q-value to bias toward repeating the previous choice.
- Computes the probability of the observed action using a softmax, accumulates the log-likelihood, and returns its negative.

### Diagram

```mermaid
flowchart TD
    A[Unpack params & init Q-values] --> B[For each trial]
    B --> C[Update Q-values]
    C --> D[Compute hybrid Q]
    D --> E[Add stickiness]
    E --> F[Softmax & log-likelihood]
    F --> G[Accumulate loglik]
    G --> H[Return -loglik]




Starting at trial 10 is typically done to exclude early trials that may be affected by practice effects, initial learning, or unstable behavior. The first few trials often do not reflect stable decision-making, as subjects are still getting familiar with the task. By starting at trial 10, the model fits only the data where the subject's choices are more likely to reflect their true learning and strategy, improving the reliability of parameter estimates. If your experiment does not have a practice phase or you want to include all trials, you can adjust this threshold.