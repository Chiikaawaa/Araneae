# Araneae

Araneae is a Physics-Informed Neural Network (PINN) implemented from scratch in C++, without using machine learning frameworks such as PyTorch or LibTorch.

The project focuses on solving a damped nonlinear pendulum problem by combining a custom neural network, automatic differentiation, and the governing physics equation.

## Features

* Custom automatic differentiation engine
* Support for first and second derivatives
* Custom MLP implementation
* `Trident` type for tracking value, first derivative, and second derivative
* Physics-based loss function
* Initial condition loss
* Training using gradient descent
* Nonlinear damped pendulum equation

## Project Structure

```text
Araneae/
├── Autograd.h    # Automatic differentiation engine
├── MLP.h         # Neuron, Layer and MLP implementation
└── pendulum.cpp  # Pendulum physics and training loop
```

## How It Works

The neural network takes time `t` as its input and predicts the angular displacement `θ(t)`.

The physics loss is based on the damped pendulum equation:

```text
θ'' + γθ' + ω²θ = 0
```

where:

* `θ` is the angular displacement
* `θ'` is the angular velocity
* `θ''` is the angular acceleration
* `γ` is the damping coefficient
* `ω²` is the angular-frequency term

The network is trained by minimizing a combination of the physics loss and the initial-condition loss.

```text
Total Loss = Physics Loss + λ × Initial Condition Loss
```

The current implementation uses a `1 → 64 → 32 → 1` MLP architecture and trains it using gradient descent.

## Automatic Differentiation

`Autograd.h` implements a small computational graph system using `Value` objects. Each value stores its numerical value, gradient, previous nodes, and a backward function.

The `Trident` class extends this idea by keeping track of:

```text
value
first derivative
second derivative
```

This allows the PINN to directly obtain the derivatives required by the differential equation.

## Requirements

* C++ compiler with C++17 or later support
* Standard C++ library

No external ML or numerical computing libraries are required.

## Building

Using `g++`:

```bash
g++ pendulum.cpp -o pendulum
```

Run with:

```bash
./pendulum
```

On Windows PowerShell:

```powershell
g++ pendulum.cpp -o pendulum
.\pendulum.exe
```

## Current Status

The basic PINN pipeline is implemented:

* Automatic differentiation
* Higher-order derivatives
* MLP
* Physics residual
* Initial conditions
* Gradient-based training

The project is currently focused on improving training stability and validating the PINN solution against a numerical solver.
