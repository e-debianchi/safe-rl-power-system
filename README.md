# Safe Reinforcement Learning for Power System Control

This repository contains the code and experiments developed for the Master's Thesis project:

**“A Safe Soft Actor-Critic Controller for Power Systems”**

Department of Computer, Control and Management Engineering  
Sapienza University of Rome  
Academic Year: 2023–2024  
**Candidate:** Emanuele De Bianchi  
**Advisor:** Alessandro Giuseppi  
**Co-advisor:** Danilo Menegatti  

## 📌 Project Overview

This project explores the use of **reinforcement learning (RL)** to design a controller capable of **stabilizing a power system** under **cyber-physical attacks** while **guaranteeing safety** at all times. It builds upon the **Soft Actor-Critic (SAC)** algorithm, which is modified to incorporate **Control Barrier Functions (CBFs)** to enforce safety constraints during learning and execution.

---

## ⚙️ Problem Setting

### ✅ Objective
- Stabilize the power system under normal and perturbed (attack) conditions.
- Ensure system safety is maintained throughout training and deployment.

### ⚡ System
- Linearized model of a power system (e.g., WSCC 9-bus system).
- Attacks modeled as exogenous disturbances with limited information.

### 🚨 Challenge
- Standard RL methods may violate safety constraints during exploration.
- Need for **fast, model-free** control that is **safe by design**.

---

## 🧠 Methodology

### 🧱 Controller Architecture
- **Actor Network**: Proposes actions based on observations.
- **Critic Network**: Evaluates value of actions.
- **Safety Layer (CBF)**: Modifies actions to enforce safety constraints via a Quadratic Program (QP) solver.

### 🔐 Safety Layer
- Represent safety constraints as smooth state-dependent functions, called Control Barrier Functions (CBFs).
- Define a *safe set* in the state space.
- Correct actions by solving a QP to stay within this set.

### 🧮 Algorithm: Safe Soft Actor-Critic (Safe-SAC)
- SAC with modified entropy regularization and off-policy updates.
- Safety-corrected actions during training and evaluation.
- Gradient backpropagation through the QP solution to enable **safe learning**.
- Attack quickly learned through a Gaussian Process (GP) Regressor *in the early stages of training*.

> 📄 **Citation**  
> Emam, Y., Notomista, G., Glotfelter, P., Kira, Z., Egerstedt, M. (2022).  
> *Safe Reinforcement Learning Using Robust Control Barrier Functions*.  
> [IEEE Robotics and Automation Letters PP(99):1-8](http://dx.doi.org/10.1109/LRA.2022.3216996)  

---

## 🧪 Experiments

### Simulated Environments
- **Toy example** for interpretability and debugging.
- **WSCC 9-bus system** for realistic power system dynamics.

### Attacks Simulated
- Non-destabilizing and destabilizing cyber-physical disturbances.
- Low disclosure but high disruption power.

### Results
- **Safe-SAC** outperforms a naive safe SAC in maintaining safety, while vanilla SAC fails to guarantee the safety.
- Learns a stable policy under perturbations.
- Robust to disturbances without prior attack model knowledge.
- Demonstrates **no training violations** when CBF correction is active.

---

## 🛠️ Installation

### Dependencies
- Python 3.8+
- PyTorch
- NumPy, SciPy, Matplotlib
- Gymnasium
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) (for pytorch standard SAC implementation)
- [cvxpylayers](https://github.com/cvxgrp/cvxpylayers) (for differentiable QP solver)
- [gpytorch](https://github.com/cornellius-gp/gpytorch) (for GP Regressor)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📊 Reproducibility

- Scripts are provided to reproduce all figures from the thesis.
- Random seeds can be set for reproducibility.
- Environments are modular and extendable.