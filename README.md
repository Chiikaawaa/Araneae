# Araneae

Araneae is a C++ learning project exploring the implementation of **automatic differentiation, neural networks, numerical ODE solvers, and Physics-Informed Neural Networks (PINNs)** from scratch.

The project uses a custom computational graph to track values and gradients, extends it to propagate first and second derivatives, and uses these components to train a small neural network to approximate the solution of a damped pendulum differential equation.

## Features

* Custom reverse-mode automatic differentiation engine using `std::shared_ptr`
* Basic arithmetic and mathematical operations including:

  * Addition, subtraction, multiplication, division
  * `exp`, `tanh`, `pow`, `sin`, and `cos`
* First-order gradient computation through a computational graph
* `Trident` type for carrying:

  * Function value
  * First derivative
  * Second derivative
* Simple feed-forward MLP implementation with configurable layers
* `tanh` activation for hidden layers
* Random parameter initialization
* Fourth-order Runge-Kutta (RK4) solver for comparison
* Physics-Informed Neural Network for solving a damped pendulum ODE

## Project Structure

```text
Araneae/
├── Autograd.h       # Automatic differentiation and Trident
├── MLP.h            # Neuron, Layer, and MLP implementation
├── RK4solver.cpp    # Numerical RK4 solution of the pendulum
└── pendulum.cpp     # PINN training implementation
```

## How It Works

The project represents computations as a graph of `Value` objects. Each value stores its numerical data, gradient, parent nodes, and a backward function. Calling `backward()` builds a topological ordering of the graph and propagates gradients through it.

On top of this, `Trident` stores a value along with its first and second derivatives. This allows the neural network to directly construct quantities such as the second derivative required by a second-order differential equation.

The MLP operates on `Trident` values, allowing derivatives to be propagated through the network while evaluating the model. Hidden layers use `tanh`, while the final layer is linear.

For the PINN experiment, the network is trained to satisfy the damped pendulum equation while also matching the initial angle and angular velocity. The loss consists of a physics residual loss and an initial-condition loss.

An independent RK4 implementation is included to numerically solve the same pendulum system and provide a conventional numerical-method reference.

## Example Setup

The current experiment uses an MLP with architecture:

```text
1 → 64 → 32 → 1
```

The PINN samples time points from the simulation interval and minimizes the differential-equation residual together with the initial-condition error.

## Goals

This project is primarily a learning and experimentation project. The main goals are to understand:

* How reverse-mode automatic differentiation works internally
* How computational graphs can be implemented in C++
* How higher-order derivatives can be propagated
* How neural networks can be used to approximate solutions to differential equations
* How PINNs compare with traditional numerical solvers such as RK4

## Future Work

Possible extensions include:

* More robust memory management for computational graphs
* Additional activation functions
* Optimizers such as Adam
* Better training and loss scaling
* Comparison of PINN predictions against RK4 solutions
* Support for higher-order derivatives
* Applying the framework to additional ODEs and PDEs

## Status

Work in progress. This project is being developed primarily as a hands-on exploration of automatic differentiation, neural networks, and scientific computing in C++.
