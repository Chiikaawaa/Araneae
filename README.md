<div align="center">

```
   _____  .______        ___      .__   __.  _______     ___       _______
  /  _  \ |   _  \      /   \     |  \ |  | |   ____|  /   \     |   ____|
 /  /_\  \|  |_)  |    /  ^  \    |   \|  | |  |__    /  ^  \    |  |__
/  /   \  |      /    /  /_\  \   |  . `  | |   __|  /  /_\  \   |   __|
\  \___/  |  |\  \   /  _____  \  |  |\   | |  |____/  _____  \  |  |____
 \_______/|__| \__\ /__/     \__\ |__| \__| |_______/__/     \__\ |_______|
```

# Araneae

**A Physics-Informed Neural Network framework for nonlinear pendulum dynamics — built from scratch in C++**

[![C++23](https://img.shields.io/badge/C%2B%2B-23-blue?logo=cplusplus&logoColor=white)](https://en.cppreference.com/w/cpp/23)
[![OpenMP](https://img.shields.io/badge/OpenMP-parallel-green?logo=openmp)](https://www.openmp.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## What Is This?

**Araneae** is an end-to-end, header-only C++ framework that:

1. **Builds its own automatic differentiation engine** — a reverse-mode tape with fused dot-product nodes (`Autograd.h`)
2. **Implements a `Trident` dual-number type** that tracks a function and its first *and* second derivatives simultaneously, enabling analytic derivative computation through the network
3. **Trains a Multilayer Perceptron (MLP)** to satisfy the nonlinear damped pendulum ODE — no training data, just physics
4. **Benchmarks the PiNN against a classic RK4 numerical integrator** in a Jupyter notebook

The name *Araneae* (the order of spiders) reflects the web-like computation graph woven by the autograd tape.

---

## The Physics

The system models a **nonlinear damped pendulum** (uniform rod, length $2r$, pivot at top):

$$I\ddot{\theta} + C\dot{\theta} + Mgr\sin(\theta) = 0$$

Normalised by $\omega_0 = \sqrt{Mgr/I}$, the network learns $\theta(\tau)$ where $\tau = \omega_0 t$:

$$\frac{d^2\theta}{d\tau^2} + \frac{\gamma}{\omega_0}\frac{d\theta}{d\tau} + \sin(\theta) = 0$$

The **physics loss** penalises the residual of this ODE at randomly sampled collocation points; the **initial condition loss** pins $\theta(0) = \theta_0$ and $\dot{\theta}(0) = \omega_0$.

| Parameter | Value |
|-----------|-------|
| Mass $M$ | 1.0 kg |
| Rod half-length $r$ | 0.2 m |
| Damping $C$ | 0.05 |
| Initial angle $\theta_0$ | $\pi/3$ rad |
| Initial angular velocity $\omega_0$ | 0 rad/s |
| Training oscillations | 4 |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Araneae Stack                    │
├─────────────────────────────────────────────────────┤
│  Jupyter Notebook  ← RK4 vs PiNN comparison plots  │
├───────────────┬─────────────────────────────────────┤
│  PiNN_run.cpp │  Inference with saved model         │
│  train.cpp    │  OpenMP-parallel training loop      │
│  RK4solver.cpp│  Reference numerical integrator     │
├───────────────┴─────────────────────────────────────┤
│  pendulum.h   │  Physics loss & IC loss             │
│  MLP.h        │  Neuron / Layer / MLP               │
│  Adam.h       │  Adam optimiser (cosine LR decay)   │
│  ModelSL.h    │  Binary model save / load           │
│  Parameter_dial.h │  All hyperparameters + encoding │
├─────────────────────────────────────────────────────┤
│  Autograd.h   │  Tape · Value · Trident             │
└─────────────────────────────────────────────────────┘
```

---

## Key Components

### `Autograd.h` — The Tape

A flat, arena-style reverse-mode tape:

- **`basic` nodes** store data, grad, up to two parents, and their local Jacobians
- **`fused` nodes** are batch dot-products (`w·x + b`) handled in a single tape entry — avoids $O(n)$ individual multiply nodes per neuron
- **`tape::freeze()` / `tape::reset()`** — after the model is built, the parameter segment is frozen; each forward pass appends *activation* nodes only, then `reset()` shrinks the tape back to the frozen mark — zero allocation overhead per training step
- **`thread_local tape Tape`** — each OpenMP thread owns its own tape, enabling lock-free parallel gradient accumulation

### `Trident` — Second-Order Dual Numbers

A `Trident` bundles $(f,\, f',\, f'')$ and propagates through every arithmetic operator and activation via the **chain and product rules**, so the network's second derivative $\ddot{\theta}$ is available analytically without a second backward pass.

```cpp
// A Trident wrapping time t carries:
//   val  = t,  de1 = 1,  de2 = 0
Trident t(Value::create(tau), Value::create(1.0), Value::create(0.0));
Trident theta = net(encode(t))[0];
// theta.de2 is d²θ/dτ² — used directly in the physics residual
```

### `MLP.h` — Network

- Architecture: **1 → [32, 32, 32] → 1** with `tanh` hidden activations and linear output
- Input is **Fourier-encoded** time (4 frequencies → 9 features) for better spectral learning
- Operates on `std::vector<Trident>` — derivatives flow through every layer for free

### `Adam.h` — Optimiser

Standard Adam with **exponential learning-rate schedule** from `lr_start = 1e-2` to `lr_end = 1e-5` over 35 000 iterations:

$$\text{lr}(i) = \text{lr\_start} \times \left(\frac{\text{lr\_end}}{\text{lr\_start}}\right)^{i/N}$$

### `train.cpp` — Parallel Training

Each iteration:
1. Sample a random batch of collocation times
2. Scatter to OpenMP threads — each thread runs its own tape in parallel
3. Accumulate gradients across threads
4. Adam step on the shared parameter vector
5. Save model binary + loss CSV

```bash
# Compile
g++ -std=c++23 -O2 -march=native -fopenmp -DNDEBUG train.cpp -o train

# Run with 8 threads, 50 000 iterations
./train 8 50000
```

### `RK4solver.cpp` — Numerical Reference

Classic fourth-order Runge–Kutta on the same ODE system:

$$Y_{n+1} = Y_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

Outputs `rk4_results.csv` for direct comparison against `pendulum_results_NN.csv`.

```bash
g++ -std=c++23 -O2 -march=native -DNDEBUG RK4solver.cpp -o RK4solver
./RK4solver
```

---

## Getting Started

### Prerequisites

| Tool | Minimum Version |
|------|----------------|
| GCC / Clang | C++23 support |
| OpenMP | 4.5+ |
| Python + Jupyter | For notebook analysis |
| NumPy, Pandas, Matplotlib | Notebook dependencies |

### Build & Run

```bash
# 1. Clone
git clone <repo-url>
cd Araneae

# 2. Train the PiNN
g++ -std=c++23 -O2 -march=native -fopenmp -DNDEBUG train.cpp -o train
./train            # uses all CPU threads, 35 000 iterations by default
# → writes: modelmodel, loss.csv

# 3. Run the PiNN for inference
g++ -std=c++23 -O2 -march=native -DNDEBUG PiNN_run.cpp -o PiNN_run
./PiNN_run
# → writes: pendulum_results_NN.csv

# 4. Run the RK4 reference
g++ -std=c++23 -O2 -march=native -DNDEBUG RK4solver.cpp -o RK4solver
./RK4solver
# → writes: rk4_results.csv

# 5. Compare in the notebook
jupyter notebook RK4_vs_PiNN_pendulum.ipynb
```

---

## Results

The notebook overlays θ(t), ω(t), and mechanical energy E(t) for both solvers. A well-trained network should:

- Track the RK4 angle trajectory within a few percent
- Conserve energy consistently (the PiNN has no numerical dissipation from time-stepping)
- Show smoother curves due to its implicit continuous-time representation

<div align="center">

*Run the notebook to generate your own comparison plots.*

</div>

---

## File Map

```
Araneae/
├── Autograd.h            ← Tape, Value, Trident
├── MLP.h                 ← Neuron, Layer, MLP
├── Adam.h                ← Adam optimiser
├── ModelSL.h             ← Model serialisation
├── pendulum.h            ← Physics & IC loss
├── Parameter_dial.h      ← Hyperparameters, Fourier encoding
│
├── train.cpp             ← OpenMP training loop
├── PiNN_run.cpp          ← Inference / CSV export
├── RK4solver.cpp         ← Numerical reference solver
│
├── modelmodel            ← Saved model weights (binary)
├── loss.csv              ← Training loss history
├── rk4_results.csv       ← RK4 trajectory
├── pendulum_results_NN.csv  ← PiNN trajectory
│
└── RK4_vs_PiNN_pendulum.ipynb  ← Comparison notebook
```

---

## Design Choices

| Decision | Rationale |
|----------|-----------|
| Flat tape (arena) over DAG with `shared_ptr` | Cache-friendly, zero allocation overhead per step after freeze |
| Fused dot-product nodes | Reduces tape size by ~$n_{\text{in}}$ nodes per neuron |
| `thread_local` tape | Lock-free parallel gradients — each thread backtracks its own segment |
| `Trident` dual numbers | Avoids a second backward pass for $d^2\theta/d\tau^2$ |
| Fourier encoding | Helps the network learn oscillatory solutions from a low-frequency bias |
| Binary model format | Minimal serialisation overhead; no external library dependency |

---

<div align="center">

*Built as an art project in physics-informed machine learning.*
*Every line of the ML stack — autodiff, optimiser, network — written by hand.*

</div>

