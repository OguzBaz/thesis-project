
---
# Model-Free vs. Model-Based RL in the Two-Step Task (Based on Collabra, 2020)

This document summarizes and compares the model-free and model-based reinforcement learning (RL) strategies used in the two-step task, based on the Collabra (2020) study: [Link to study](https://online.ucpress.edu/collabra/article/6/1/17213/114338/).

---

## Task Summary: Two-Step Decision Task

- **Stage 1**: Choose between two actions (A1, A2).
- **Transition**: Leads to one of two states (S1 or S2) via a common (70%) or rare (30%) transition.
- **Stage 2**: Choose again, receive reward or not.

---


## Hybrid Reinforcement Model
---

## 🧮 Hybrid Model Example: Two-Step Task

This example demonstrates how the hybrid model combines model-free and model-based values to guide choices in the two-step task.

**Assumptions:**
- Learning rate: $\alpha = 0.2$
- Eligibility trace: $\lambda = 0.8$
- Hybrid weight: $w = 0.6$ (60% model-based, 40% model-free)
- Transition probabilities:
    - Spaceship A1: 70% to Planet 1 ($s_2$), 30% to Planet 2 ($s_3$)
    - Spaceship A2: 70% to Planet 2 ($s_3$), 30% to Planet 1 ($s_2$)
- Q-values before the trial:
    - $Q_{\mathrm{MF}}(s_1, a_1) = 0.5$
    - $Q_{\mathrm{MF}}(s_1, a_2) = 0.4$
    - $Q_{\mathrm{MF}}(s_2, a_1) = 0.3$
    - $Q_{\mathrm{MF}}(s_2, a_2) = 0.2$
    - $Q_{\mathrm{MF}}(s_3, a_1) = 0.6$
    - $Q_{\mathrm{MF}}(s_3, a_2) = 0.5$

### Step 1: Model-Based Value Calculation

For action $a_1$ at $s_1$:
$$
Q_{\mathrm{MB}}(s_1, a_1) = 0.7 \times \max(0.3, 0.2) + 0.3 \times \max(0.6, 0.5) = 0.7 \times 0.3 + 0.3 \times 0.6 = 0.21 + 0.18 = 0.39
$$

For action $a_2$ at $s_1$:
$$
Q_{\mathrm{MB}}(s_1, a_2) = 0.7 \times 0.6 + 0.3 \times 0.3 = 0.42 + 0.09 = 0.51
$$

### Step 2: Hybrid Value Calculation

$$
Q_{\mathrm{Hybrid}}(s_1, a) = w \cdot Q_{\mathrm{MB}}(s_1, a) + (1-w) \cdot Q_{\mathrm{MF}}(s_1, a)
$$

For $a_1$:
$$
Q_{\mathrm{Hybrid}}(s_1, a_1) = 0.6 \times 0.39 + 0.4 \times 0.5 = 0.234 + 0.2 = 0.434
$$
For $a_2$:
$$
Q_{\mathrm{Hybrid}}(s_1, a_2) = 0.6 \times 0.51 + 0.4 \times 0.4 = 0.306 + 0.16 = 0.466
$$

### Step 3: Action Selection

The hybrid model uses these $Q_{\mathrm{Hybrid}}$ values to choose between $a_1$ and $a_2$ at $s_1$.
- $Q_{\mathrm{Hybrid}}(s_1, a_1) = 0.434$
- $Q_{\mathrm{Hybrid}}(s_1, a_2) = 0.466$

**Model would pick:** $a_2$ (since $0.466 > 0.434$)

### Step 4: Q-value Update (after reward)

Suppose you pick $a_2$, transition to $s_3$, pick $a_1$ at $s_3$, and receive reward $r = 1$.

- Compute prediction error for $s_3, a_1$:
$$
\delta = r - Q_{\mathrm{MF}}(s_3, a_1) = 1 - 0.6 = 0.4
$$
- Update $Q_{\mathrm{MF}}(s_3, a_1)$:
$$
Q_{\mathrm{MF}}(s_3, a_1) \leftarrow 0.6 + 0.2 \times 0.4 = 0.6 + 0.08 = 0.68
$$
- Update $Q_{\mathrm{MF}}(s_1, a_2)$ with eligibility trace:
$$
Q_{\mathrm{MF}}(s_1, a_2) \leftarrow 0.4 + 0.2 \times 0.8 \times 0.4 = 0.4 + 0.064 = 0.464
$$

### Step 5: Updated Q-values

| State | Q(a1) | Q(a2) |
|-------|-------|-------|
| $s_1$ | 0.5   | 0.464 |
| $s_2$ | 0.3   | 0.2   |
| $s_3$ | 0.68  | 0.5   |

---

**Python code for this example:**

```python
# Hybrid model example
alpha = 0.2
lam = 0.8
w = 0.6

Q_MF = {('s1', 'a1'): 0.5, ('s1', 'a2'): 0.4,
        ('s2', 'a1'): 0.3, ('s2', 'a2'): 0.2,
        ('s3', 'a1'): 0.6, ('s3', 'a2'): 0.5}

# Model-based values
Q_MB_a1 = 0.7 * max(Q_MF[('s2', 'a1')], Q_MF[('s2', 'a2')]) + 0.3 * max(Q_MF[('s3', 'a1')], Q_MF[('s3', 'a2')])
Q_MB_a2 = 0.7 * max(Q_MF[('s3', 'a1')], Q_MF[('s3', 'a2')]) + 0.3 * max(Q_MF[('s2', 'a1')], Q_MF[('s2', 'a2')])

# Hybrid values
Q_Hybrid_a1 = w * Q_MB_a1 + (1 - w) * Q_MF[('s1', 'a1')]
Q_Hybrid_a2 = w * Q_MB_a2 + (1 - w) * Q_MF[('s1', 'a2')]

print('Hybrid Q-values:', {'a1': Q_Hybrid_a1, 'a2': Q_Hybrid_a2})
# Output: {'a1': 0.434, 'a2': 0.466}

# Suppose we pick a2, go to s3, pick a1, get reward 1
delta = 1 - Q_MF[('s3', 'a1')]
Q_MF[('s3', 'a1')] += alpha * delta
Q_MF[('s1', 'a2')] += alpha * lam * delta

print('Updated Q_MF:', Q_MF)
# Output: {('s1', 'a1'): 0.5, ('s1', 'a2'): 0.464, ('s2', 'a1'): 0.3, ('s2', 'a2'): 0.2, ('s3', 'a1'): 0.68, ('s3', 'a2'): 0.5}
```




The hybrid reinforcement learning model consists of both a **model-free** and **model-based** learning algorithm, which separately compute the value of each action <span style="font-family: 'Latin Modern Math', 'Cambria Math', 'STIX Math', 'Arial', sans-serif;">(a)</span> in each state <span style="font-family: 'Latin Modern Math', 'Cambria Math', 'STIX Math', 'Arial', sans-serif;">(s)</span> on each trial <span style="font-family: 'Latin Modern Math', 'Cambria Math', 'STIX Math', 'Arial', sans-serif;">(t)</span>.

The **model-free** algorithm computes the value of each first- and second-stage choice option based on a standard temporal difference equation:


$$
Q_{\mathrm{MF}}(s_{i,t}, a_{i,t}) = Q_{\mathrm{MF}}(s_{i,t-1}, a_{i,t-1}) + \alpha \cdot \delta_{i,t}
$$


- <span style="font-family: 'Latin Modern Math', 'Cambria Math', 'STIX Math', 'Arial', sans-serif;">$\alpha$</span>: Learning rate (free parameter capturing how quickly values are updated)
- <span style="font-family: 'Latin Modern Math', 'Cambria Math', 'STIX Math', 'Arial', sans-serif;">$\delta_{i,t}$</span>: Prediction error at trial <span style="font-family: 'Latin Modern Math', 'Cambria Math', 'STIX Math', 'Arial', sans-serif;">$t$</span>

Where the prediction error is:


$$
\delta_{i,t} = r_{i,t} - Q_{\mathrm{MF}}(s_{i,t-1}, a_{i,t-1})
$$

At the **first stage**, the prediction error is the difference between the expected value of the action selected in the first stage and the expected value of the second-stage state-action value.

At the **second stage**, the prediction error is the difference between the reward actually experienced and the expected value of the action selected in the second stage.

This second-stage prediction error is then multiplied by an **eligibility trace** ($\lambda$), a free parameter, before it is used to update first-stage state-action values:


$$
Q_{\mathrm{MF}}(s_{1,t}, a_{1,t}) \leftarrow Q_{\mathrm{MF}}(s_{1,t}, a_{1,t}) + \alpha \lambda \cdot \delta_{2,t}
$$

This allows the second-stage outcome to influence first-stage learning. This update is equivalent under the model-based learning algorithm, but the first-stage learning function differs as described below.

---

## 📊 Numerical Example: Model-Free Q-Learning in the Two-Step Task

Suppose you are at the **second stage** and just received a reward. Let's walk through a concrete example for the model-free update:

**Assumptions:**
- Learning rate $\alpha = 0.2$
- Eligibility trace $\lambda = 0.8$
- Previous Q-values:
    - $Q_{\mathrm{MF}}(s_1, a_1) = 0.5$
    - $Q_{\mathrm{MF}}(s_2, a_2) = 0.3$
- You choose $a_2$ in $s_2$ and receive reward $r = 1$

**Step 1: Compute second-stage prediction error**

$$
\delta_{2} = r - Q_{\mathrm{MF}}(s_2, a_2) = 1 - 0.3 = 0.7
$$

**Step 2: Update second-stage Q-value**

$$
Q_{\mathrm{MF}}(s_2, a_2) \leftarrow Q_{\mathrm{MF}}(s_2, a_2) + \alpha \cdot \delta_{2}
$$
$$
Q_{\mathrm{MF}}(s_2, a_2) = 0.3 + 0.2 \times 0.7 = 0.3 + 0.14 = 0.44
$$

**Step 3: Update first-stage Q-value using eligibility trace**

$$
Q_{\mathrm{MF}}(s_1, a_1) \leftarrow Q_{\mathrm{MF}}(s_1, a_1) + \alpha \lambda \cdot \delta_{2}
$$
$$
Q_{\mathrm{MF}}(s_1, a_1) = 0.5 + 0.2 \times 0.8 \times 0.7 = 0.5 + 0.112 = 0.612
$$

**Result:**
- $Q_{\mathrm{MF}}(s_2, a_2)$ is now **0.44**
- $Q_{\mathrm{MF}}(s_1, a_1)$ is now **0.612**

---

**Python code for this update:**

```python
# Model-free Q-learning update with eligibility trace
alpha = 0.2
lam = 0.8
Q = {('s1', 'a1'): 0.5, ('s2', 'a2'): 0.3}
reward = 1.0

# Second-stage update
delta2 = reward - Q[('s2', 'a2')]
Q[('s2', 'a2')] += alpha * delta2

# First-stage update with eligibility trace
Q[('s1', 'a1')] += alpha * lam * delta2

print(Q)
# Output: {('s1', 'a1'): 0.612, ('s2', 'a2'): 0.44}
```

---

## 🔄 Continuing Example: Next Trial Update

Suppose on the **next trial** you again choose $a_2$ in $s_2$ and receive a reward $r = 0$ (no reward). Let's update the Q-values using the new values from the previous example:

**Current Q-values:**
- $Q_{\mathrm{MF}}(s_1, a_1) = 0.612$
- $Q_{\mathrm{MF}}(s_2, a_2) = 0.44$

**Step 1: Compute new second-stage prediction error**
$$
\delta_{2} = r - Q_{\mathrm{MF}}(s_2, a_2) = 0 - 0.44 = -0.44
$$

**Step 2: Update second-stage Q-value**
$$
Q_{\mathrm{MF}}(s_2, a_2) \leftarrow Q_{\mathrm{MF}}(s_2, a_2) + \alpha \cdot \delta_{2}
$$
$$
Q_{\mathrm{MF}}(s_2, a_2) = 0.44 + 0.2 \times (-0.44) = 0.44 - 0.088 = 0.352
$$

**Step 3: Update first-stage Q-value using eligibility trace**
$$
Q_{\mathrm{MF}}(s_1, a_1) \leftarrow Q_{\mathrm{MF}}(s_1, a_1) + \alpha \lambda \cdot \delta_{2}
$$
$$
Q_{\mathrm{MF}}(s_1, a_1) = 0.612 + 0.2 \times 0.8 \times (-0.44) = 0.612 - 0.0704 = 0.5416
$$

**Result after second trial:**
- $Q_{\mathrm{MF}}(s_2, a_2)$ is now **0.352**
- $Q_{\mathrm{MF}}(s_1, a_1)$ is now **0.5416**

**Python code for this second update:**

```python
# Continuing the example: next trial, reward = 0
reward = 0.0
delta2 = reward - Q[('s2', 'a2')]
Q[('s2', 'a2')] += alpha * delta2
Q[('s1', 'a1')] += alpha * lam * delta2
print(Q)
# Output: {('s1', 'a1'): 0.5416, ('s2', 'a2'): 0.352}
```

---

## 🧠 2. Model-Based Value Estimation


$$
Q_{\mathrm{MB}}(s_{1,t}, a_{j,t}) = P(s_2|s_1,a_j)\max_{a \in \{a_A, a_B\}} Q_{\mathrm{TD}}(s_2,a) + P(s_3|s_1,a_j)\max_{a \in \{a_A, a_B\}} Q_{\mathrm{TD}}(s_3,a)
$$

- Computes expected values for each first-stage action based on known transition probabilities.
- **$Q_{\mathrm{TD}}(s,a)$**: Learned Q-values at second-stage states.

This reflects planning and task-structure awareness.

---

## 🎯 3. Hybrid Action Selection (Softmax Rule)


$$
P(a_{i,t} = a | s_{i,t}) = \frac{\exp[\beta_{\mathrm{MF}} Q_{\mathrm{MF}}(s_{i,t}, a) + \beta_{\mathrm{MB}} Q_{\mathrm{MB}}(s_{i,t}, a) + p \cdot \mathrm{rep}(a)]}{\sum_{a'} \exp[\beta_{\mathrm{MF}} Q_{\mathrm{MF}}(s_{i,t}, a') + \beta_{\mathrm{MB}} Q_{\mathrm{MB}}(s_{i,t}, a') + p \cdot \mathrm{rep}(a')]}
$$

- **$\beta_{\mathrm{MF}}$**: Influence of model-free value
- **$\beta_{\mathrm{MB}}$**: Influence of model-based value
- **$p$**: Bias parameter for action repetition ($\mathrm{rep}(a)$)

---

## Summary of Parameters

| Parameter           | Description                          |
|--------------------|--------------------------------------|
| $\alpha$           | Learning rate                        |
| $\delta$           | Reward prediction error               |
| $\beta_{\mathrm{MF}}$ | Softmax temperature for MF values   |
| $\beta_{\mathrm{MB}}$ | Softmax temperature for MB values   |
| $p$                | Action repetition bias               |

---

## Suggested Use in Python

Use this `.md` file in Jupyter Notebooks, or render it using Markdown packages in Python for clean documentation.

Let me know if you want the PyMC3 or `rlssm` implementation next.
