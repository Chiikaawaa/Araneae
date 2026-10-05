#ifndef PARAMETER_DIAL_H
#define PARAMETER_DIAL_H

#include<cmath>
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
    int batch_size = 50;
    double h = 0.01;
    double loss_scaling = 100.0;
    int iterations = 20000;

    double learning_rate = 1e-3;
    double start_lr = 1e-3;
    double end_lr = 5 * 1e-5;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;
};

#endif
