# Araneae

## About

Araneae is a physics informed neural network (PiNN) that learns the motion of a damped compound pendulum with zero training data. All it gets is the governing ODE and the initial conditions. No strings attached, though the pendulum has one.

Everything is written from scratch in C++:

- a reverse mode autodiff engine (scalar, graph based, micrograd style)
- a second order truncated Taylor forward mode on top of it, so the network can give you theta, dtheta/dt and d2theta/dt2 in one pass
- a Karpathy style MLP built on those two
- a training loop with mini batches and Adam

The spider name is because the autodiff graph is basically a web of nodes. That is the whole pun budget for this paragraph. also because both spider and pendulum swing...

## Equation of Pendulum

A uniform rod of length 2r swings about one end, with viscous damping at the pivot:

```
d^2 theta / dt^2 + gamma * d theta / dt + angfreqsqr * sin(theta) = 0
```

where

- `I = (1/3) * M * (2r)^2` is the moment of inertia of the rod about the pivot
- `gamma = C / I` is the damping rate
- `angfreqsqr = M * g * r / I` is the square of the angular frequency (small angle, no damping)
- `M` is the mass of the rod (kg)
- `g` is gravity (m/s^2)
- `r` is the distance from the pivot to the center of mass, so the rod is `2r` long (m)
- `C` is the damping coefficient, torque_damping = -C * dtheta/dt
- `theta` is the angle from the vertical, `omega = dtheta/dt` is the angular velocity

Defaults: M = 1, g = 9.81, r = 0.2, C = 0.05, theta0 = pi/3, omega0 = 0.

### Non dimensionalising

Let `w = sqrt(angfreqsqr)` and `tau = time * w`. Then `d/dt = w * d/dtau` and the equation becomes

```
angfreqsqr * d2theta/dtau2 + w * gamma * dtheta/dtau + angfreqsqr * sin(theta) = 0
```

Divide everything by `angfreqsqr` and you get the form the code actually uses:

```
d2theta/dtau2 + (gamma / w) * dtheta/dtau + sin(theta) = 0
```

The network takes `tau` as input and outputs `theta`. Derivatives with respect to `tau` come out of the forward mode. To get back to real units, `omega = dtheta/dtau * w`.

## How it learns

The network never sees a solution. It sees the ODE residual:

```
residual = d2theta/dtau2 + (gamma / w) * dtheta/dtau + sin(theta)
```

If theta(tau) is a true solution, the residual is zero everywhere. So the loss is:

```
physics_loss = mean over batch of residual^2
ic_loss      = (theta(0) - theta0)^2 + (dtheta/dtau(0) - omega0 / w)^2
loss         = lambda_ph * physics_loss + lambda_ic * ic_loss
```

The physics loss is evaluated at random time points. The IC loss pins the starting angle and velocity, otherwise "theta = 0 forever" is a perfectly valid answer and the network would happily take it.

### The autodiff bit

`Value` is a scalar node holding `data`, `grad`, and a backward closure. `backward()` does a topological sort and walks it in reverse.

`Trident` holds three `Value`s: the value, the first derivative and the second derivative with respect to the input. Each op (`+`, `*`, `/`, `Exp`, `Tanh`, `Pow`, `Sin`, `Cos`) propagates all three with the product and chain rules. Because those derivative terms are themselves `Value` nodes in the graph, the loss, which contains d2theta/dtau2, can be backpropagated all the way to the weights. Forward mode for the input derivatives, reverse mode for the weight gradients.

## Network

- Input: 9 features (see below)
- Hidden layers: 32, 32, 32 with tanh
- Output: 1 neuron, linear, this is theta
- 2465 parameters
- Weights start uniform in [-0.5, 0.5], biases at 0
- The RNG is seeded from `random_device`, so no two training runs are the same

## Fourier feature encoding

Instead of feeding raw `tau`, the network gets:

```
[ tau, sin(tau), cos(tau), sin(2 tau), cos(2 tau), ... , sin(4 tau), cos(4 tau) ]
```

`n_freq = 4`, so 1 + 2*4 = 9 inputs.

Why: plain MLPs with tanh are bad at fitting oscillating functions of a raw scalar input. They learn smooth low frequency stuff first and struggle with waves. The pendulum is a wave, so we hand the network sines and cosines up front. Since the input is `tau`, the natural oscillation sits near frequency 1 (damping is light, so the period in tau is close to 2 pi), and the higher harmonics cover the distortion from the large swing (60 degrees is not small angle). The raw `tau` stays in the list so the network can still build the decay envelope from damping.

The encoding is itself written with `Trident` ops, so derivatives flow through it for free.

## Adam and mini batches

**Mini batch.** Every iteration draws `batch_size` (100) fresh random times uniformly from `[0, 4 periods]` and averages the squared residual over them. Fresh points each step means the network cannot memorise a fixed grid, and each step is cheaper than evaluating a dense grid.

**Adam.** Per parameter it keeps a running average of the gradient (`m`, decay `beta1`) and of the squared gradient (`v`, decay `beta2`), corrects both for their startup bias, and steps by `lr * m_hat / (sqrt(v_hat) + epsilon)`. Parameters with noisy or large gradients get smaller steps, quiet ones get bigger steps.

**Learning rate schedule.** Exponential decay from `lr_start = 1e-2` to `lr_end = 1e-5` over all iterations, so it swings hard early and settles down at the end.

## Files

| File | Purpose |
|---|---|
| `Autograd.h` | `Value` (reverse mode) and `Trident` (value + 1st + 2nd derivative forward mode) |
| `MLP.h` | `Neuron`, `Layer`, `MLP` built on `Trident`, plus `zero_grad` |
| `ModelSL.h` | Save and load the weights as raw binary |
| `Parameter_dial.h` | All the knobs, the global RNG, and the Fourier `encode` function |
| `pendulum_train.cpp` | Training: physics + IC loss, mini batches, Adam. Writes `pendulum` and `loss.csv` |
| `PiNN_run.cpp` | Loads `pendulum` and evaluates the network on a uniform time grid. Writes `pendulum_results_NN.csv` |
| `RK4solver.cpp` | Classic RK4 solution of the same ODE, used as ground truth. Writes `rk4_results.csv` |
| `RK4_vs_PiNN.ipynb` | Plots network vs RK4 for theta and omega |

## Requirements

- A C++17 compiler (structured bindings are used in `Autograd.h`). A recent g++ works with the commands below as is. Older compilers need `-std=c++17`.
- `RK4solver.cpp` uses unicode identifiers (`θ`, `ω`). On an older compiler, rename them to plain ASCII.
- Python for the notebook: `pandas`, `matplotlib`, `numpy`, `seaborn` (it is imported). `scipy` only if you want extra analysis.

## Build and run

Order matters: `PiNN_run` needs the `pendulum` file that training writes, and it reads it from the current directory.

```
g++ -O3 -march=native -DNDEBUG pendulum_train.cpp -o pendulum_train
./pendulum_train
g++ -O3 -march=native -DNDEBUG PiNN_run.cpp -o PiNN_run
./PiNN_run
g++ -O3 -march=native -DNDEBUG RK4solver.cpp -o RK4solver
./RK4solver
```

`pendulum_train` prints `physics_loss`, `ic_loss` and elapsed time every 10 iterations (not seconds). Delete that print if you want to save a bit of IO.

Training is slow-ish by design: every collocation point builds its own full autodiff graph, and there are 100 of them per iteration for 35000 iterations.

## Outputs

| File | Written by | Contents |
|---|---|---|
| `pendulum` | `pendulum_train` | Binary weights: a `uint64` parameter count, then each parameter as a `double`. The architecture in `PiNN_run.cpp` must match, or loading fails with a parameter count mismatch |
| `loss.csv` | `pendulum_train` | `iteration,physics_loss,ic_loss`, one row per iteration |
| `pendulum_results_NN.csv` | `PiNN_run` | `time,theta,omega` from the network, omega in rad/s |
| `rk4_results.csv` | `RK4solver` | `time,theta_r,omega_r` from RK4 with step 0.01, including the t = 0 row |

## Configuration

Everything lives in `Parameter_dial.h`.

**`Pendulum_params`**

| Name | Default | Meaning |
|---|---|---|
| `M` | 1.0 | Mass |
| `g` | 9.81 | Gravity |
| `r_len` | 0.2 | Pivot to center of mass, rod length is twice this |
| `C` | 0.05 | Damping coefficient |
| `theta0` | pi/3 | Initial angle |
| `omega0` | 0.0 | Initial angular velocity |
| `I`, `angfreqsqr`, `gamma`, `time_period` | derived | Computed from the above. `time_period` is the small angle damped period |

**`Training_params`**

| Name | Default | Meaning |
|---|---|---|
| `total_oscillations` | 4.0 | Training window in periods |
| `batch_size` | 100 | Collocation points per iteration |
| `h` | 0.01 | Time step of the evaluation grid in `PiNN_run` |
| `lambda_ph`, `lambda_ic` | 1.0, 1.0 | Loss weights |
| `iterations` | 35000 | Adam steps |
| `lr_start`, `lr_end` | 1e-2, 1e-5 | Learning rate decay endpoints |
| `beta1`, `beta2`, `epsilon` | 0.9, 0.999, 1e-8 | Adam constants |

`n_freq` (default 4) sets the number of Fourier frequencies. The network sizes `{32, 32, 32, 1}` are hardcoded in both `pendulum_train.cpp` and `PiNN_run.cpp`. Change one, change both, and retrain.

## Things worth knowing

- The network trains on 4 periods but `PiNN_run` and `RK4solver` run for 5. That last stretch is extrapolation, so expect the network to take a swing and a miss there.
- `time_period` is the small angle formula. At 60 degrees the real period is longer, so "4 periods" is a rough window size and not an exact count of oscillations.
- `RK4solver.cpp` hardcodes `h = 0.01`. It does not read `trn.h`, so keep them in sync if you want rows to line up in the notebook.
- The grid in `PiNN_run` is `total_time * i / ns` with an integer `ns`, so its spacing is only close to `h`. The two CSVs have slightly different row counts and times, so compare them by plotting, not by row index.
- No fixed seed, so loss curves differ between runs.
