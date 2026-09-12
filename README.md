# Araneae

A from-scratch C++ implementation of a lightweight automatic differentiation engine, built as the foundation for my Physics-Informed Neural Network (PINN) project.

## Overview

Araneae implements a small computational-graph based automatic differentiation system in C++, inspired by how frameworks such as PyTorch build and traverse computational graphs.

The goal is to understand and implement the underlying mechanics myself rather than relying on an existing deep-learning/autodiff library.

## Features

- Computational graph using `std::shared_ptr`
- Automatic graph construction during mathematical operations
- Backpropagation through the computational graph
- First-order derivatives
- Second-order derivatives
- Support for common mathematical operations:
  - Addition
  - Subtraction
  - Multiplication
  - Division
  - Exponentials
  - Powers
  - `sin`
  - `cos`
  - `tanh`
- Custom `Trident` class for carrying:
  - Function value
  - First derivative
  - Second derivative

## Why?

This project is part of my work toward implementing a **Physics-Informed Neural Network completely from scratch in C++**, without using PyTorch, LibTorch, or dedicated numerical-method libraries.

The automatic differentiation engine is intended to provide the derivatives required by the physics residual of the PINN.

## Current Direction

Araneae is being developed as a foundation for:

1. Automatic differentiation
2. Neural-network forward propagation
3. First and second derivatives through the network
4. Physics-informed loss calculation
5. Solving differential equations with a PINN

The eventual target is a PINN for a **damped compound pendulum**.

## Technologies

- C++
- Standard Library
- `std::shared_ptr`
- Computational graphs
- Automatic differentiation
- Calculus

## Status

🚧 **Work in Progress**

The automatic differentiation and second-order derivative components are currently being developed and tested before being integrated into the full PINN implementation.
