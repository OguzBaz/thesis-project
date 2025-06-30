# Reinforcement Learning Models in the Two-Step Task

This document provides clear explanations of the three main reinforcement learning (RL) strategies used in the two-step decision task: model-free, model-based, and hybrid RL. Use this as a reference for your cognitive modeling or behavioral analysis projects.

---

## 1. Model-Free RL

**Model-free RL** learns the value of actions directly from experience, without considering the structure of the environment. It uses a temporal difference (TD) learning rule:

```math
Q_{\mathrm{MF}}(s, a) \leftarrow Q_{\mathrm{MF}}(s, a) + \alpha \cdot [r - Q_{\mathrm{MF}}(s, a)]
```

- $Q_{\mathrm{MF}}(s, a)$: Model-free value of action $a$ in state $s$
- $\alpha$: Learning rate (how quickly values are updated)
- $r$: Reward received


**Key features:**
- Updates are based only on received rewards (no planning)
- Fast and simple, but can be inflexible


---

## 2. Model-Based RL

**Model-based RL** uses knowledge of the environment's structure (e.g., transition probabilities) to plan ahead. It computes the value of each action by considering possible future states and their values:

```math
Q_{\mathrm{MB}}(s_1, a) = \sum_{s_2} P(s_2 | s_1, a) \cdot \max_{a'} Q_{\mathrm{MF}}(s_2, a')
```

- $P(s_2 | s_1, a)$: Probability of reaching state $s_2$ from $s_1$ by taking action $a$
- $Q_{\mathrm{MF}}(s_2, a')$: Model-free value at the second stage

**Key features:**
- Uses a model of the environment to plan
- Flexible and adaptive, but computationally demanding

---

## 3. Hybrid RL Model

**Hybrid RL** combines model-free and model-based values to guide choices. The hybrid value is a weighted sum:

```math
Q_{\mathrm{Hybrid}}(s_1, a) = w \cdot Q_{\mathrm{MB}}(s_1, a) + (1 - w) \cdot Q_{\mathrm{MF}}(s_1, a)
```

- $w$: Weighting parameter ($0 \leq w \leq 1$) between model-based and model-free control

**Key features:**
- Captures both habitual (model-free) and goal-directed (model-based) behavior
- The parameter $w$ reflects the balance between the two systems

---

## 4. Action Selection (Softmax Rule)

Choices are made using a softmax function, which converts value estimates into choice probabilities:

```math
P(a | s) = \frac{\exp(\beta Q(s, a))}{\sum_{a'} \exp(\beta Q(s, a'))}
```

- $\beta$: Inverse temperature (higher = more deterministic choices)

---

## Summary Table

| Model         | Description                                 |
|--------------|---------------------------------------------|
| Model-Free   | Learns from rewards, no planning            |
| Model-Based  | Plans using environment structure           |
| Hybrid       | Mixes model-free and model-based strategies |

---

**References:**
- Daw et al. (2011). Model-based influences on humans' choices and striatal prediction errors. *Neuron, 69*(6), 1204–1215.
- [Collabra (2020) study](https://online.ucpress.edu/collabra/article/6/1/17213/114338/)
