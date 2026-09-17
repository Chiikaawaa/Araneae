#include "Autograd.h"
#include "MLP.h"
#include <cmath>
#include <vector>

using namespace std;

struct PhysicsResult {
    Value::sptr physics_loss;
    Value::sptr ic_loss;
};
PhysicsResult pendulum(const MLP& net, double h, const Trident& t_end, double gamma, double angfreqsqr, double theta0, double omega0) {
    Value::sptr physics_loss = Value::create(0.0);
    int n_steps = t_end.val->data / h;
    for (int step = 0; step < n_steps; step++) {
        double time = step * h;
        Trident t(Value::create(time/10.0), Value::create(1.0/10.0), Value::create(0.0));
        vector<Trident> t_vec{t};
        Trident theta = net(t_vec)[0];
        Value::sptr r = theta.de2 + gamma * theta.de1 + angfreqsqr * theta.val;
        physics_loss  = physics_loss + r*r;
    }
    physics_loss = physics_loss / n_steps;
    Trident t0(Value::create(0.0), Value::create(1.0), Value::create(0.0));
    std::vector<Trident> t0_vec{t0};
    Trident ic_pred = net(t0_vec)[0];

    Value::sptr ic_loss = pow(ic_pred.val - theta0, 2.0) + pow(ic_pred.de1 - omega0, 2.0);

    return {physics_loss, ic_loss};
}

int main() {
    MLP net(1, {64, 32, 1});
    double M = 1.0, g = 9.81, r_len = 0.2;
    double theta0 = M_PI / 3.0;
    double omega0 = 0.0;
    double I = 0.0533;
    double C = 0.2;
    double angfreqsqr = M * g * r_len / I;
    double gamma = C / I;

    Trident t_end(Value::create(10.0), Value::create(0.0), Value::create(0.0));
    double learning_rate = 0.000001;
    double loss_scaling  = 15;
    double h             = 0.01;
    for(int i = 0; i < 1000; i++) {
        auto result = pendulum(net, h, t_end, gamma, angfreqsqr, theta0, omega0);
        Value::sptr loss = result.physics_loss + loss_scaling * result.ic_loss;

        net.zero_grad();
        loss->backward();
        for(auto &p: net.parameters()) {
            p.val->data -= learning_rate * p.val->grad;
        }
        cout<<i<<" "<<"physics_loss = "<<result.physics_loss->data<<" ic_loss = "<<result.ic_loss->data<<endl;
    }
    return 0;
}
