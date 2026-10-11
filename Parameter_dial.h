#ifndef PARAMETER_DIAL_H
#define PARAMETER_DIAL_H

#include "Autograd.h"

#include <cmath>
#include <iostream>
#include <vector>

constexpr double pi = 3.141592653589793238462643;

struct Pendulum_params {
    double M = 1.0;
    double g = 9.81;
    double r_len = 0.2;
    double C = 0.05;

    double theta0 = pi / 3.0;
    double omega0 = 0.0;

    double I;
    double angfreqsqr;
    double gamma;
    double time_period;

    Pendulum_params() {
        I = (1.0 / 3.0) * M * (2.0 * r_len) * (2.0 * r_len);
        angfreqsqr = M * g * r_len / I;
        gamma = C / I;
        time_period = 2.0 * pi / std::sqrt(angfreqsqr - std::pow(gamma / 2.0, 2.0));
    }
};

struct Training_params {
    double total_oscillations = 4.0;
    int batch_size = 100;
    double h = 0.01;
    double lambda_ic = 1.0;
    double lambda_ph = 1.0;
    int iterations = 35000;

    double learning_rate = 1e-2;
    double lr_start = learning_rate;
    double lr_end = 1e-5;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;
};

inline Pendulum_params pnd;
inline Training_params trn;

constexpr int n_freq = 4;
constexpr int n_inputs = 1 + 2 * n_freq;

inline std::vector<Trident> encode(const Trident& t) {
    std::vector<Trident> f;
    f.emplace_back(t);
    for (int k = 1; k <= n_freq; k++){
        Trident a = t * double(k);
        f.push_back(a.Sin());
        f.push_back(a.Cos());
    }
    return f;
}

#endif
