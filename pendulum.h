#ifndef PENDULUM_H
#define PENDULUM_H

#include "Autograd.h"
#include "MLP.h"
#include "Parameter_dial.h"

struct PhysicsResults {
    Value physics_loss;
    Value ic_loss;
};

inline PhysicsResults pendulum (const MLP& net,
    const std::vector<double>& times,
    int lo, int hi, bool with_ic) {
    Value physics_loss = Value::create(0.0);
    double w0 = sqrt(pnd.angfreqsqr);

    for(int b = lo; b < hi; b++) {
        double tau = times[b] * w0;
        Trident t(Value::create(tau), Value::create(1.0), Value::create(0.0));
        Trident theta = net(encode(t))[0];

        Value r = theta.de2 + pnd.gamma * theta.de1 / w0 + sin(theta.val);
        physics_loss = physics_loss + r*r;
    }
    physics_loss = physics_loss / trn.batch_size;

    Value ic_loss = Value::create(0.0);
    if(with_ic) {
        Trident t0(Value::create(0.0), Value::create(1.0), Value::create(0.0));
        Trident ic_pred = net(encode(t0))[0];
        ic_loss = pow(ic_pred.val - pnd.theta0, 2.0) + pow(ic_pred.de1 - (pnd.omega0 / w0), 2.0);
    }

    return {physics_loss, ic_loss};
}

#endif
