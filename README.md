# Araneae 🕷️⏱️

## About

**Araneae** is a physics-informed neural network (PiNN) that learns the non-linear motion of a damped compound pendulum with **zero training data**. All it gets is the governing ODE and the initial conditions. No strings attached, though the pendulum has one.

Everything is written from scratch in C++:

- A **flat tape-based reverse-mode autodiff engine** with contiguous memory, zero dynamic heap allocations during training, and fused dot-product nodes.
- A **second-order truncated Taylor forward mode** on top of it (`Trident`), letting the network evaluate $\theta$, $\frac{d\theta}{d\tau}$, and $\frac{d^2\theta}{d\tau^2}$ in a single unified forward pass.
- A **Karpathy-style micrograd-inspired MLP** built on `Trident` with fused weight-bias linear layers.
- A **parallel training loop** with OpenMP multi-threading, thread-local tapes, mini-batch collocation, and Adam optimization.

The spider name? Because the autodiff graph is basically a web of nodes. That is the whole pun budget for this paragraph. Also because both spiders and pendulums swing...

---

## What's New: The Speed Odyssey

### From v1 to v2: Getting Our Act on Tape
- **v1 (Shared Pointer Spaghetti):** Built the autodiff graph out of `Value` objects managed by `std::shared_ptr`, one heap allocation per node, each with a backward `std::function` lambda closure, and a recursive topological sort on every `backward()`. It worked, but it spent more time talking to `malloc` than doing math.
- **v2 (The Tape):** Threw all that out for a flat, pre-reserved `std::vector<basic>` (`tape Tape`). `Value` became a lightweight 4-byte `uint32_t` index. Because operations are pushed in execution order, the tape is **already in topological order**. Backward is just a linear reverse sweep down the array. At the end of each iteration, `Tape.reset()` truncates the vector back to the frozen parameter mark in $O(1)$ time. Graph recycled, zero leaks.

### Beyond v2: Fused Nodes & OpenMP Multi-Threading
1. **Fused Linear Operations (1.47x speedup):**
   Why push 30 separate addition and multiplication nodes onto the tape for a single neuron when you can compute $\mathbf{w} \cdot \mathbf{x} + b$ in one shot? We introduced `NodeType::Fused` and `push_fused()`. Backward sweeps through the inputs and weights in a tight, cache-friendly loop (`x.grad += w.data * n.grad`, `w.grad += x.data * n.grad`). Less tape, less overhead, 1.47x pure speedup (see `speedup.png`).
2. **OpenMP Multi-Threading (`thread_local` Tape):**
   Remember when earlier versions warned that *"two threads sharing the Tape would be a bad time"*? We took that personally. We made `Tape` **`thread_local`**! Each OpenMP thread now gets its own private tape. Collocation batches are sliced evenly across CPU threads, each thread computes physics residuals and backward passes lock-free, and gradients are accumulated cleanly before the Adam step.
3. **Modular Architecture:**
   The training harness is now modular:
   - `pendulum.h`: Physics ODE residuals and multi-threaded collocation slice evaluation.
   - `Adam.h`: Dedicated Adam optimizer class encapsulating moment buffers, bias correction, and learning rate scheduling.
   - `train.cpp`: Unified training driver with CLI arguments for both thread count and iterations.

### Benchmarks

| Version | Time per 100 iterations | Time per iteration | Total Time (25,000 iters) |
|---|---|---|---|
| **v1 (Initial build)** | ~500 s | ~5.0 s | ~35 hours (pack a sleeping bag) |
| **v1 (Optimized pointers)** | ~50 s | ~0.5 s | ~3.5 hours |
| **v2 (Flat Tape)** | ~1.7 s | ~0.017 s | ~7 minutes |
| **Latest (Fused + OpenMP)** | **~0.22 s – 0.35 s** | **~2.5 ms – 3.5 ms** | **~55 – 80 seconds** ⚡ |

*(Measured with 8 OpenMP threads on a modern x86 CPU with `-O3 -march=native -DNDEBUG`)*

---

## Equation of Pendulum

A uniform rigid rod of length $2r$ swings about one pivot end, subject to gravity and viscous damping:

```
d^2 theta / dt^2 + gamma * d theta / dt + angfreqsqr * sin(theta) = 0
```

Where:

- $I = \frac{1}{3} M (2r)^2$ is the moment of inertia of the rod about the pivot.
- $\gamma = \frac{C}{I}$ is the damping rate.
- $\text{angfreqsqr} = \frac{M g r}{I}$ is the square of the small-angle undamped angular frequency.
- $M$ is the mass of the rod (kg).
- $g$ is gravitational acceleration ($9.81\ \text{m/s}^2$).
- $r$ is the distance from pivot to center of mass (the rod is $2r$ long).
- $C$ is the damping coefficient ($\tau_{\text{damping}} = -C \dot{\theta}$).
- $\theta$ is the angle from vertical; $\omega = \dot{\theta}$ is the angular velocity.

**Defaults:** `M = 1.0`, `g = 9.81`, `r = 0.2`, `C = 0.05`, $\theta_0 = \pi/3$ ($60^\circ$), $\omega_0 = 0.0$.

---

## Non-Dimensionalising

Why drag physical constants around when you can scale time?

Let $\omega_n = \sqrt{\text{angfreqsqr}}$ and dimensionless time $\tau = t \cdot \omega_n$. By the chain rule, $\frac{d}{dt} = \omega_n \frac{d}{d\tau}$, transforming the equation into:

```
angfreqsqr * d2theta/dtau2 + omega_n * gamma * dtheta/dtau + angfreqsqr * sin(theta) = 0
```

Divide through by $\text{angfreqsqr}$ and you get the elegant dimensionless ODE the code actually solves:

```
d2theta/dtau2 + (gamma / omega_n) * dtheta/dtau + sin(theta) = 0
```

The neural network takes $\tau$ as input and outputs $\theta(\tau)$. Derivatives with respect to $\tau$ emerge automatically from the forward-mode `Trident`. To retrieve real-world units: $\omega = \frac{d\theta}{d\tau} \cdot \omega_n$.

---

## How It Learns

The network never sees a pre-computed trajectory. It only sees the ODE residual:

```
residual = d2theta/dtau2 + (gamma / omega_n) * dtheta/dtau + sin(theta)
```

If $\theta(\tau)$ is a true solution, the residual is zero everywhere. Thus, the total loss is:

```
physics_loss = mean over batch of (residual^2)
ic_loss      = (theta(0) - theta0)^2 + (dtheta/dtau(0) - omega0 / omega_n)^2
loss         = lambda_ph * physics_loss + lambda_ic * ic_loss
```

- **Physics loss:** Sampled across uniformly distributed random collocation points in $[0, 4\text{ periods}]$.
- **Initial condition (IC) loss:** Anchors the starting angle and velocity. Otherwise, $\theta(\tau) = 0$ for all time is a trivial residual-free equilibrium, and the network would happily take it (minimum energy and maximum laziness, much like a Sunday morning).

---

## The Autodiff Engine

### Tape and Fused Nodes
- **`Tape` and `Value`:** A `Value` is simply an index into a thread-local `tape`. Standard scalar arithmetic operations push `basic` nodes recording inputs and local derivatives.
- **Fused Dot-Products:** Fully connected layers call `push_fused(xi, wi, bias_id)`. Instead of blowing up the tape with dozens of intermediate sum and product nodes, it packs the layer computation into a `fused` entry, evaluating both the forward activation and exact analytical backprop gradients in a vectorized loop.
- **`Trident` (Forward-Mode Taylor Arithmetic):** Holds three `Value` indices: value, 1st derivative, and 2nd derivative with respect to $\tau$. Standard operators (`+`, `-`, `*`, `/`, `Exp`, `Tanh`, `Pow`, `Sin`, `Cos`) propagate all three simultaneously using product and chain rules. Because derivative terms are themselves tape nodes, the physics loss (which includes $\ddot{\theta}$) backpropagates directly into the network weights.

### Tape Lifecycle in Multi-Threaded Training

```
Build MLP on main thread   -> Parameters are the first n_params nodes on tape
Tape.freeze()              -> Lock the boundary (bmark)
Main loop:
    #pragma omp parallel:
        Clone frozen parameter prefix to thread_local Tape (if needed)
        Evaluate chunk of collocation batch via pendulum.h
        loss.backward()    -> Reverse sweep down thread-local Tape
        Save thread gradients into gbuf[threadID]
        Tape.reset()       -> Truncate back to mark; graph gone in O(1)
    Sum gradients across threads: g = sum(gbuf)
    adam.step(params, g)   -> Parameter update via Adam.h
```

---

## Network Architecture

- **Input:** 9 Fourier features derived from $\tau$.
- **Hidden Layers:** 3 dense layers of 32 neurons each with `Activation::Tanh`.
- **Output:** 1 linear neuron (`Activation::linear`) predicting $\theta$.
- **Parameters:** 2,465 weights and biases.
- **Initialization:** Weights uniform in $[-0.5, 0.5]$, biases at 0.0.
- **RNG:** Seeded from `std::random_device`.

### Fourier Feature Encoding

Standard MLPs with `tanh` suffer from spectral bias—they learn low frequencies easily and struggle with waves. But a pendulum is literally a wave. 

Instead of feeding raw $\tau$, the network receives:

```
[ tau, sin(tau), cos(tau), sin(2 tau), cos(2 tau), sin(3 tau), cos(3 tau), sin(4 tau), cos(4 tau) ]
```

With `n_freq = 4`, we supply 9 features:
- Low harmonics capture the dominant oscillation (near frequency 1).
- Higher harmonics capture non-linear distortion caused by large initial swing angles ($60^\circ$ is far beyond small-angle territory).
- The raw $\tau$ term lets the network capture the exponential decay envelope from damping.
- The encoding is evaluated with `Trident` operations, so all input derivatives propagate through the harmonics for free.

---

## Files

| File | Purpose |
|------|---------|
| `Autograd.h` | Tape storage (`tape`, `basic`, `fused`), lightweight `Value` indices, and `Trident` forward-mode Taylor arithmetic. |
| `MLP.h` | `Neuron`, `Layer`, and `MLP` architectures with fused dot products and instant gradient zeroing. |
| `Adam.h` | Modular Adam optimizer implementation with learning rate schedule and moment tracking. |
| `pendulum.h` | Physics residual and initial condition loss functions with multi-threaded chunking support. |
| `ModelSL.h` | Binary serialization (`save_model`, `load_model`) for model weights. |
| `Parameter_dial.h` | Physical constants (`pnd`), training hyperparams (`trn`), optimizer knobs (`dlo`), and Fourier encoding. |
| `train.cpp` | Main training program: OpenMP parallel loop, loss logging to `loss.csv`, and model checkpointing to `modelmodel`. |
| `PiNN_run.cpp` | Inference evaluator: loads `modelmodel`, evaluates over time grid with tape recycling, and exports `pendulum_results_NN.csv`. |
| `RK4solver.cpp` | Classical 4th-order Runge-Kutta numerical ground-truth solver, exporting `rk4_results.csv`. |
| `RK4_vs_PiNN_pendulum.ipynb` | Jupyter Notebook comparing PiNN predictions vs RK4 ground truth (angle, angular velocity, phase portrait). |
| `speedup.png` | Benchmark plot demonstrating the 1.47x speedup from weight/bias fusion. |

---

## Requirements

- **C++ Compiler:** Modern C++ (C++20 or C++23 recommended; GCC 12+, Clang 15+, or MSVC 2022) with OpenMP support (`-fopenmp`).
- **Python (for notebook):** `python >= 3.8` with `pandas`, `numpy`, `matplotlib`, and `seaborn`.

---

## Build and Run

> [!NOTE]
> Order matters: `PiNN_run` expects the trained binary weights file `modelmodel` written by `train`.

### 1. Compile & Train

Compile with OpenMP and optimization enabled:

```bash
g++ -std=c++23 -O3 -march=native -fopenmp -DNDEBUG train.cpp -o train
```

Run training:
```bash
./train
```

*Want custom threads and iteration counts? Pass them directly via CLI:*
```bash
./train 8 25000         # ./train [nthreads] [iterations]
```

*(Keep `-DNDEBUG` enabled for real runs; assertions in `Autograd.h` and `MLP.h` are great for debugging the tape, but they slow down training).*

### 2. Run Inference (Evaluation)

Compile and run the inference executable:

```bash
g++ -std=c++23 -O3 -march=native -DNDEBUG PiNN_run.cpp -o PiNN_run
./PiNN_run
```

*(If you see `"Load kela model, thamb pudhcha karto"`, don't panic. That's Marathi for "Model loaded, hold your horses while I do the rest!")*

### 3. Generate RK4 Ground Truth

Compile and run the Runge-Kutta integrator:

```bash
g++ -std=c++23 -O3 -march=native -DNDEBUG RK4solver.cpp -o RK4solver
./RK4solver
```

### 4. Compare & Visualize

Launch Jupyter and open `RK4_vs_PiNN_pendulum.ipynb` to plot the neural network's predictions against the numerical RK4 integration.

---

## Outputs

| File | Written by | Contents |
|------|-----------|----------|
| `modelmodel` | `train` | Binary weights: `uint64` parameter count followed by sequential `double` values. |
| `loss.csv` | `train` | `iteration,physics_loss,ic_loss` across training. |
| `pendulum_results_NN.csv` | `PiNN_run` | `time,theta,omega` predicted by the network. |
| `rk4_results.csv` | `RK4solver` | `time,theta_r,omega_r` from RK4 with step `h = 0.01`. |

---

## Configuration

Everything lives in `Parameter_dial.h`.

**`Pendulum_params` (`pnd`)**

| Name | Default | Meaning |
|------|---------|---------|
| `M` | `1.0` | Mass of the rod (kg) |
| `g` | `9.81` | Acceleration due to gravity ($\text{m/s}^2$) |
| `r` | `0.2` | Pivot to center of mass (rod length is $2r$) |
| `C` | `0.05` | Damping coefficient |
| `theta0` | `pi / 3.0` | Initial displacement ($60^\circ$ in radians) |
| `omega0` | `0.0` | Initial angular velocity ($\text{rad/s}$) |
| `I`, `angfreqsqr`, `gamma`, `time_period` | *derived* | Computed automatically from the physical parameters above |

**`Training_params` (`trn`)**

| Name | Default | Meaning |
|------|---------|---------|
| `total_oscillations` | `4.0` | Time span in small-angle periods |
| `batch_size` | `100` | Collocation points per iteration |
| `h` | `0.01` | Evaluation and integration grid step size |
| `lambda_ph`, `lambda_ic` | `1.0, 1.0` | Loss term weights |
| `iterations` | `25000` | Adam optimization steps (overridable via CLI) |

**`DLoptimiser` (`dlo`)**

| Name | Default | Meaning |
|------|---------|---------|
| `learning_rate` / `lr_start` | `1e-2` | Initial learning rate |
| `lr_end` | `1e-5` | Final decayed learning rate |
| `beta1`, `beta2`, `epsilon` | `0.9, 0.999, 1e-8` | Adam hyperparameters |

---

## Things Worth Knowing

- **Threads and Tapes:** The tape is `thread_local`. Every OpenMP thread gets its own contiguous memory vector. Threads never lock, never share tape nodes, and never wait on each other during the forward or backward sweeps.
- **Tape Recycling:** `Value` is an index into the tape. Calling `Tape.reset()` rewinds the vector to the mark, making any temporary intermediate `Value` invalid. Network parameters are safe because they live permanently below the frozen mark.
- **Inference Tape Hygiene:** `PiNN_run` calls `Tape.freeze()` before the evaluation loop and `Tape.reset()` inside the loop, preventing the tape from growing uncontrollably over large grids.
- **Period Nuance:** `time_period` uses the small-angle linear formula. At $60^\circ$, non-linear effects lengthen the true physical period, so "4 periods" is an approximate window rather than an exact integer count of swings.
- **Model Name:** The checkpoint is named `modelmodel`. Why? Because it's so good we named it twice.
- **Zero Training Data:** The network has never seen a labeled dataset, sensor measurement, or simulation log. It learns to swing purely by listening to the laws of physics and minimizing the ODE residual—proof that with enough C++, autograd tape, and coffee, a neural net can figure out gravity all by itself.
