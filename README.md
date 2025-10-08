# 🧠 IncentRL: Bayesian Adaptation of Preference Gaps in Reinforcement Learning

> **ICLR 2026 Submission**  
> A cognitively inspired reinforcement learning framework that unifies external rewards and internal incentives through Bayesian adaptation of preference gaps.

---

## 🌌 Overview

**IncentRL** introduces a new paradigm for **reward shaping** by integrating cognitive principles from neuroscience and the Free Energy Principle (FEP) into reinforcement learning.

Instead of fixing the trade-off between extrinsic rewards and intrinsic motivation, IncentRL **treats the incentive coefficient** ($\beta$) **as a Bayesian random variable**, allowing automatic adaptation across tasks and environments.

The framework combines:
- **External rewards** (task-specific)
- **Internal incentives** based on **KL divergence** between predicted and preferred outcome distributions
- **Bayesian adaptation** of the incentive weight $\beta$ using an online posterior update mechanism

---
## 🧩 Key Idea

Traditional RL agents optimize for external rewards, often struggling in sparse or delayed-reward settings.  
**IncentRL** adds an **internal alignment term** inspired by cognitive dissonance and predictive coding:

$$
\mathcal{L}_{\text{IncentRL}} = -\mathbb{E}_{(s,a)\sim \pi}\Big[ r_{\text{ext}}(s,a) - \beta \, D_{KL}(p(o|s,a)\,\|\,q(o|s)) \Big]
$$

where:

- $$\( p(o|s,a) \)$$: predicted outcome distribution  
- $$\( q(o|s) \$$): preferred outcome distribution  
- $$\( \beta \sim \text{Beta}(\alpha, \gamma) \)$$: dynamically sampled incentive weight updated via Bayesian posterior  

This forms an **adaptive incentive shaping** mechanism that bridges:

- Exploration–exploitation trade-offs  
- Extrinsic–intrinsic reward unification  
- Cognitive–computational symmetry *(dopamine RPE ↔ Bayesian belief updating)*

---

## 🧮 Theoretical Foundations

IncentRL draws inspiration from:

- **Free Energy Principle (Friston, 2010)** — minimizing surprise through prediction–preference alignment  
- **Self-Information Theory (SIT)** — incentives as information gain over expected outcomes  
- **Renormalization and category-theoretic abstraction** — interpreting policy updates as entropy-reducing morphisms  
- **Inverse diffusion analogy** — from random exploratory action to coherent intentional behavior  

These theoretical lenses connect incentive shaping to **entropy minimization** and **psychological coherence**.

---

## 🚀 Features

- **Bayesian β-adaptation:** removes manual tuning of reward trade-offs  
- **KL-based incentive shaping:** links prediction and preference distributions  
- **Cognitively grounded:** connects to dopamine RPE and FEP  
- **Task-agnostic:** works across sparse-reward and exploration-heavy domains  
- **Simple integration:** can wrap around existing RL algorithms (PPO, DQN, A2C, etc.)

---

## 🧠 Architecture
