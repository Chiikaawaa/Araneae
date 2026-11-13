# Araneae

## About

Araneae is a physics informed neural network (PiNN) that learns the motion of a damped compound pendulum with zero training data. All it gets is the governing ODE and the initial conditions. No strings attached, though the pendulum has one.

Everything is written from scratch in C++:

- a reverse mode autodiff engine (scalar, tape based, micrograd style)
- a second order truncated Taylor forward mode on top of it, so the network can give you `theta`, `dtheta/dt` and `d2theta/dt2` in one pass
- a Karpathy style MLP built on those two
- a training loop with mini batches and Adam

The spider name is because the autodiff graph is basically a web of nodes. That is the whole pun budget for this paragraph. Also because both spider and pendulum swing...

## What's new in v2

v1 built the autodiff graph out of `Value` objects held by shared pointers, one heap allocation per node, each with a backward closure, and a topological sort on every `backward()`. v2 throws all of that out.

- **The graph is a tape.** Every node lives in one global `std::vector<node>` (`Tape`), reserved up front. A `node` is plain data: the value, the gradient, and up to two parents, each stored as an index plus the local derivative with respect to that parent.
- **`Value` is just an index.** It is a 4 byte `uint32_t` into the tape. No shared pointers, no refcounting, no closures, no per node `new`.
- **`backward()` is one reverse sweep.** Nodes are pushed in creation order, so the tape is already topologically sorted. There is nothing to sort: walk from the output index back down to the frozen mark, push `local_derivative * grad` into each parent, and skip nodes whose gradient is zero.
- **The tape is recycled.** After the model is built, `Tape.freeze()` marks where the parameters end. At the end of each iteration `Tape.reset()` truncates the vector back to that mark, which throws away the whole iteration's graph in one go. Parameters live below the mark and survive.
- **Parameters are contiguous.** The MLP records `first_id` and `n_params`, so `zero_grad()` is a simple loop over a slice of the tape.
- Training now defaults to 25000 iterations (was 35000) and takes an optional command line override: `./pendulum_train 5000`.
- Progress prints every 100 iterations (was 10), and `elapsed` is now in seconds.
- Training also prints the total time and a "Model saved successfully" message at the end.

### Speed

| Version | Time per iteration |
|---------|--------------------|
| v1, first working build | ~5 s |
| v1, after optimising | ~0.5 s |
| v2 (tape) | ~0.017 s (1.7 s per 100 iterations) |

That is roughly 30x faster than the best v1, and a full 25000 iteration run lands in the neighbourhood of 7 minutes instead of hours. Your numbers will depend on your machine.

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
- `C` is the damping coefficient, `torque_damping = -C * dtheta/dt`
- `theta` is the angle from the vertical, `omega = dtheta/dt` is the angular velocity

Defaults: `M = 1`, `g = 9.81`, `r = 0.2`, `C = 0.05`, `theta0 = pi/3`, `omega0 = 0`.

## Non dimensionalising

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

If `theta(tau)` is a true solution, the residual is zero everywhere. So the loss is:

```
physics_loss = mean over batch of residual^2
ic_loss      = (theta(0) - theta0)^2 + (dtheta/dtau(0) - omega0 / w)^2
loss         = lambda_ph * physics_loss + lambda_ic * ic_loss
```

The physics loss is evaluated at random time points. The IC loss pins the starting angle and velocity, otherwise "theta = 0 forever" is a perfectly valid answer and the network would happily take it.

## The autodiff bit

**`Tape` and `Value`.** A `Value` is an index into the global `Tape`. Each op (`+`, `-`, `*`, `/`, `exp`, `tanh`, `pow`, `sin`, `cos`) pushes one node holding the result, plus its one or two parents and the local derivative for each, computed on the spot. `backward()` seeds the output node's gradient with 1 and sweeps the tape in reverse, accumulating `local_derivative * grad` into the parents.

**`Trident`** holds three `Value`s: the value, the first derivative and the second derivative with respect to the input. Each op (`+`, `-`, `*`, `/`, `Exp`, `Tanh`, `Pow`, `Sin`, `Cos`) propagates all three with the product and chain rules. Because those derivative terms are themselves tape nodes, the loss, which contains `d2theta/dtau2`, can be backpropagated all the way to the weights. Forward mode for the input derivatives, reverse mode for the weight gradients.

**Lifecycle of the tape in training:**

```
build MLP          -> parameters are the first n_params nodes on the tape
Tape.freeze()      -> remember where the parameters end (the "mark")
each iteration:
    build graph    -> residuals and loss push nodes above the mark
    zero_grad()    -> zero the parameter gradients
    backward()     -> reverse sweep from the loss down to the mark
    Adam update    -> params[j].data() -= ...
    Tape.reset()   -> truncate back to the mark, graph gone
```

## Network

- Input: 9 features (see below)
- Hidden layers: 32, 32, 32 with `tanh`
- Output: 1 neuron, linear, this is `theta`
- 2465 parameters
- Weights start uniform in `[-0.5, 0.5]`, biases at 0
- The RNG is seeded from `random_device`, so no two training runs are the same

### Fourier feature encoding

Instead of feeding raw `tau`, the network gets:

```
[ tau, sin(tau), cos(tau), sin(2 tau), cos(2 tau), ... , sin(4 tau), cos(4 tau) ]
```

`n_freq = 4`, so `1 + 2*4 = 9` inputs.

Why: plain MLPs with `tanh` are bad at fitting oscillating functions of a raw scalar input. They learn smooth low frequency stuff first and struggle with waves. The pendulum is a wave, so we hand the network sines and cosines up front. Since the input is `tau`, the natural oscillation sits near frequency 1 (damping is light, so the period in `tau` is close to 2 pi), and the higher harmonics cover the distortion from the large swing (60 degrees is not small angle). The raw `tau` stays in the list so the network can still build the decay envelope from damping.

The encoding is itself written with `Trident` ops, so derivatives flow through it for free.

## Adam and mini batches

**Mini batch.** Every iteration draws `batch_size` (100) fresh random times uniformly from `[0, 4 periods]` and averages the squared residual over them. Fresh points each step means the network cannot memorise a fixed grid, and each step is cheaper than evaluating a dense grid.

**Adam.** Per parameter it keeps a running average of the gradient (`m`, decay `beta1`) and of the squared gradient (`v`, decay `beta2`), corrects both for their startup bias, and steps by `lr * m_hat / (sqrt(v_hat) + epsilon)`. Parameters with noisy or large gradients get smaller steps, quiet ones get bigger steps.

**Learning rate schedule.** Exponential decay from `lr_start = 1e-2` to `lr_end = 1e-5` over all iterations, so it swings hard early and settles down at the end.

## Files

| File | Purpose |
|------|---------|
| `Autograd.h` | `tape`/`node` storage, `Value` (reverse mode, an index into the tape) and `Trident` (value + 1st + 2nd derivative forward mode) |
| `MLP.h` | `Neuron`, `Layer`, `MLP` built on `Trident`, plus `zero_grad` |
| `ModelSL.h` | Save and load the weights as raw binary |
| `Parameter_dial.h` | All the knobs, the global RNG, and the Fourier `encode` function |
| `pendulum_train.cpp` | Training: physics + IC loss, mini batches, Adam. Writes `pendulum` and `loss.csv` |
| `PiNN_run.cpp` | Loads `pendulum` and evaluates the network on a uniform time grid. Writes `pendulum_results_NN.csv` |
| `RK4solver.cpp` | Classic RK4 solution of the same ODE, used as ground truth. Writes `rk4_results.csv` |
| `RK4_vs_PiNN.ipynb` | Plots network vs RK4 for `theta` and `omega` |

## Requirements

- A C++17 compiler (the code relies on `inline` variables). The training source's own build comment uses `-std=c++23`(completely unnecessary, c++17 works), any a recent `g++` works with the commands below as is.
- `RK4solver.cpp` uses unicode identifiers (`θ`, `ω`). On an older compiler, rename them to plain ASCII.
- Python for the notebook: `pandas`, `matplotlib`. `scipy` only if you want extra analysis.

## Build and run

Order matters: `PiNN_run` needs the `pendulum` file that training writes, and it reads it from the current directory.

```bash
g++ -std=c++23 -O3 -march=native -DNDEBUG pendulum_train.cpp -o pendulum_train
./pendulum_train            # 25000 iterations default, or: ./pendulum_train -mention number of training iterations-
g++ -std=c++23 -O3 -march=native -DNDEBUG PiNN_run.cpp -o PiNN_run
./PiNN_run
g++ -std=c++23 -O3 -march=native -DNDEBUG RK4solver.cpp -o RK4solver
./RK4solver
```

Keep `-DNDEBUG` for real runs. Without it the `assert`s in `MLP.h` and `Autograd.h` stay on, which is useful while debugging the tape but not free.

`pendulum_train` prints `physics_loss`, `ic_loss` and elapsed seconds every 100 iterations. Delete that print if you want to save a bit of IO.

Training is still the expensive part: every collocation point builds its own full autodiff graph on the tape, and there are 100 of them per iteration. The difference is that v2 does it in about 17 ms per iteration instead of 500.

## Outputs

| File | Written by | Contents |
|------|-----------|----------|
| `pendulum` | `pendulum_train` | Binary weights: a `uint64` parameter count, then each parameter as a `double`. The architecture in `PiNN_run.cpp` must match, or loading fails with a parameter count mismatch |
| `loss.csv` | `pendulum_train` | `iteration,physics_loss,ic_loss`, one row per iteration |
| `pendulum_results_NN.csv` | `PiNN_run` | `time,theta,omega` from the network, `omega` in rad/s |
| `rk4_results.csv` | `RK4solver` | `time,theta_r,omega_r` from RK4 with step 0.01, including the `t = 0` row |

## Configuration

Everything lives in `Parameter_dial.h`.

**`Pendulum_params`**

| Name | Default | Meaning |
|------|---------|---------|
| `M` | 1.0 | Mass |
| `g` | 9.81 | Gravity |
| `r` | 0.2 | Pivot to center of mass, rod length is twice this |
| `C` | 0.05 | Damping coefficient |
| `theta0` | pi/3 | Initial angle |
| `omega0` | 0.0 | Initial angular velocity |
| `I`, `angfreqsqr`, `gamma`, `time_period` | derived | Computed from the above. `time_period` is the small angle damped period |

**`Training_params`**

| Name | Default | Meaning |
|------|---------|---------|
| `total_oscillations` | 4.0 | Training window in periods |
| `batch_size` | 100 | Collocation points per iteration |
| `h` | 0.01 | Time step of the evaluation grid in `PiNN_run`, and the RK4 step |
| `lambda_ph`, `lambda_ic` | 1.0, 1.0 | Loss weights |
| `iterations` | 25000 | Adam steps (overridable from the command line) |

**`DLoptimiser`**

| Name | Default | Meaning |
|------|---------|---------|
| `lr_start`, `lr_end` | 1e-2, 1e-5 | Learning rate decay endpoints |
| `beta1`, `beta2`, `epsilon` | 0.9, 0.999, 1e-8 | Adam constants |

`n_freq` (default 4) sets the number of Fourier frequencies. The network sizes `{32, 32, 32, 1}` are hardcoded in both `pendulum_train.cpp` and `PiNN_run.cpp`. Change one, change both, and retrain.

## Things worth knowing

- The network trains on 4 periods but `PiNN_run` and `RK4solver` run for 5. That last stretch is extrapolation, so expect the network to take a swing and a miss there.
- `time_period` is the small angle formula. At 60 degrees the real period is longer, so "4 periods" is a rough window size and not an exact count of oscillations.
- The grid in `PiNN_run` is `total_time * i / ns` with an integer `ns`, so its spacing is only close to `h`. The two CSVs have slightly different row counts and times, so compare them by plotting, not by row index.
- **A `Value` is only valid until the next `Tape.reset()`.** It is an index, so after a reset it points at whatever gets pushed next. Do not hold on to per iteration `Value`s across iterations. Parameters are safe because they sit below the mark.
- **Order matters on the tape.** Build the `MLP` first, then call `Tape.freeze()`, then train. `Tape.reset()` and `MLP::zero_grad()` assert that the tape was frozen (asserts are off under `-DNDEBUG`, so getting this wrong will not warn you).
- The tape is global and the code is single threaded. Two threads sharing the `Tape` would be a bad time.
- The tape is `reserve`d at `2 << 20` nodes, which is on the order of 100 MB at 48 bytes per node. A batch of 100 collocation points fills a good chunk of that. If you raise `batch_size` or widen the network the vector will simply grow, but you pay for the reallocation and the memory.
- `PiNN_run` never calls `Tape.freeze()` or `Tape.reset()`, so its tape keeps growing over the evaluation grid. That is fine for a few hundred points, and it is an easy fix if you ever evaluate many more.
