# Federated Active Inference (FAI) - FRAMEWORK

### Distributed Minimization of Variational Free Energy at Scale: Free-energy minimization alone is sufficient to drive distributed learning, convergence, and model simplification in a federated multi-agent system.

## Project Overview

This repository implements a **Federated Active Inference** framework. It bridges the gap between **Theoretical Neuroscience (The Free Energy Principle)** and **Distributed Machine Learning (Federated Learning)**.

The simulation features **1,000 decentralized agents** that collaboratively optimize a shared generative model. Unlike standard Federated Learning which minimizes frequentist loss, this system minimizes **Variational Free Energy (VFE)**, allowing agents to balance sensory accuracy with model complexity.

---

## Active inference is not just a cognitive theory — it functions as a scalable learning principle.

### 1. Active Inference & Belief Updating

Each agent functions as a variational inference engine. It maintains internal beliefs about hidden environmental states and updates them to minimize "surprise" (Free Energy).

* **Complexity:** The  divergence between the agent's posterior and its prior beliefs.
* **Accuracy:** The expected log-likelihood of observations given the current model parameters.

### 2. Bayesian Model Reduction

Agents are capable of **structural learning**. As the global model stabilizes, agents evaluate whether a "Complex" or "Simple" prior better minimizes their local Free Energy, favoring simplicity when environmental uncertainty is low.
Model compression is an emergent property of Bayesian inference — not a heuristic.

### 3. Federated Aggregation (FedAvg)

Nodes compute gradients of their local Free Energy functionals and transmit weight updates to a global server. The server aggregates these into a **Global Likelihood Matrix ()**, which is then redistributed to all agents.
Local variational objectives induce coherent global learning.

---

We demonstrate that federated populations of active inference agents self-organize toward accurate, low-complexity generative models solely through variational free-energy minimization, inducing convergence, structural simplification, and global coordination without centralized supervision.

## Features

* **Large Scale:** Simulated environment with **1,000 nodes** and **100 training rounds**.
* **Real-time Visualization:** Integrated Matplotlib dashboard showing:
* **VFE Convergence:** The downward trajectory of population surprise.
* **Model Drift:** The stabilization of the global parameter manifold.
* **Model Adoption:** The phase transition from complex to simple internal models.


* **Robust Engine:** Pure Python implementation of variational math (Softmax, KL-Divergence) with numerical stability protections.


## Dashboard Metrics

| Metric | Description | Scientific Significance |
| --- | --- | --- |
| **Avg Free Energy** | Collective surprise across the population. | Measures the "fit" of the global generative model. |
| **Global Drift** | Magnitude of change in the Likelihood Matrix. | Indicates convergence to a "Consensus Reality." |
| **Simple Model Ratio** | Percentage of agents utilizing reduced priors. | Demonstrates Occam's Razor in a variational setting. |

---

## Mathematical Implementation

The core objective function minimized by the network is defined as:

Where  represents the global weights defining the observation likelihoods shared across the federated network.

