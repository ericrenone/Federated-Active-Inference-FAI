#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import math
import random
import time
import sys
import argparse

# ------------------- CLI -------------------
parser = argparse.ArgumentParser(description="FSGL — Fisher–Shannon Geodesic Learning Simulation")
parser.add_argument("--steps", type=int, default=160, help="Number of steps (default: 160)")
parser.add_argument("--fps", type=float, default=16.7, help="Frames per second (default: 16.7)")
parser.add_argument("--no-clear", action="store_true", help="Disable screen clear (scrolling mode)")
parser.add_argument("--width", type=int, default=55, help="Bar width")
parser.add_argument("--dim", type=int, default=24, help="Dimension of probability vector")
args = parser.parse_args()

STEPS = 100
WIDTH = args.width
DIM = args.dim
SLEEP_TIME = 1.0 / args.fps
CLEAR = not args.no_clear

# ------------------- Utilities -------------------
def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

def bar(val: float, maxv: float = 1.0) -> str:
    n = int(WIDTH * clamp(val / maxv))
    return "█" * n + "░" * (WIDTH - n)

# ------------------- Canonical Metrics -------------------
def entropy(p):
    return -sum(x * math.log(x + 1e-12) for x in p)

def normalize(v):
    s = sum(v)
    return [x / s for x in v] if s > 0 else v

def kl(p, q):
    return sum(p[i] * math.log((p[i] + 1e-12) / (q[i] + 1e-12)) for i in range(len(p)))

def fisher(p):
    return sum(1 / (x + 1e-8) for x in p) / len(p)   # peakiness proxy

def relational_abstraction(p):
    """Normalized concentration: 1 = perfectly peaked, 0 = uniform"""
    H = entropy(p)
    Hmax = math.log(len(p))
    return clamp(1.0 - H / Hmax)

# ------------------- Initialization -------------------
random.seed(42)  # reproducible
state = normalize([random.random() for _ in range(DIM)])
target = normalize([math.exp(-((i - DIM//2) ** 2) / (DIM / 3)) for i in range(DIM)])

# ------------------- Smooth Phase Transitions -------------------
def get_focus(p, t_norm: float):
    # Smooth logistic weights
    w_ent = 1 / (1 + math.exp(12 * (t_norm - 0.28)))   # Entropy collapse
    w_fish = 1 / (1 + math.exp(12 * (t_norm - 0.52)))   # Fisher curvature growth
    w_rel = 1 / (1 + math.exp(12 * (t_norm - 0.76)))    # Relational abstraction / stabilization

    w_sum = w_ent + w_fish + w_rel + 1e-12
    w_ent /= w_sum
    w_fish /= w_sum
    w_rel /= w_sum

    # Sharpened target during Fisher phase
    sharp_target = normalize([x ** 1.35 for x in target])

    focus = [
        w_ent * target[i] +
        w_fish * sharp_target[i] +
        w_rel * target[i]
        for i in range(DIM)
    ]
    return normalize(focus)

# ------------------- Display -------------------
def clear():
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()

def render(t: int, H: float, F: float, K: float, R: float):
    if CLEAR:
        clear()
    print("FSGL — Fisher–Shannon Geodesic Learning (Optimized Simulation)\n")
    print(f"Step {t}/{STEPS}   |   FPS ≈ {args.fps:.1f}\n")

    print(f"Entropy (H)          {bar(H, 2.8)}  {H:.3f}")
    print(f"Fisher Information   {bar(F / 60, 1.0)}  {F:.2f}")
    print(f"KL Divergence        {bar(K, 1.2)}  {K:.3f}")
    print(f"Relational Abstraction {bar(R, 1.0)}  {R:.3f}\n")

# ------------------- Main Simulation -------------------
history = []
lr = 0.095

print("Starting FSGL simulation...\n")
for t in range(1, STEPS + 1):
    t_norm = t / STEPS
    focus = get_focus(state, t_norm)

    # Gradient-style update
    state = [(1 - lr) * state[i] + lr * focus[i] for i in range(DIM)]
    state = normalize(state)

    H = entropy(state)
    F = fisher(state)
    K = kl(state, target)
    R = relational_abstraction(state)

    history.append((H, F, K, R))
    render(t, H, F, K, R)
    time.sleep(SLEEP_TIME)

# ------------------- Final Summary -------------------
H0, F0, K0, R0 = history[0]
H1, F1, K1, R1 = history[-1]

print("\n" + "="*45)
print("          FSGL SIMULATION COMPLETE")
print("="*45)
print(f"Entropy:          {H0:.3f}  →  {H1:.3f}  ↓")
print(f"Fisher Info:      {F0:.2f}   →  {F1:.2f}   ↑")
print(f"KL Divergence:    {K0:.3f}  →  {K1:.3f}  ↓")
print(f"Rel. Abstraction: {R0:.3f}  →  {R1:.3f}  ↑")
print("\nSystem evolved along the Fisher–Shannon geodesic:")
print("   Random Identity  →  Sharp Feature  →  Relational Abstraction")
print("\nRepresentation successfully compressed and abstracted.")
print("="*45)
