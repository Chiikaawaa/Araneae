#include "Autograd.h"
#include "MLP.h"
#include <cmath>
#include <vector>
#include <fstream>
#define pi 3.141592653589793238462643


using namespace std;

struct PhysicsResult {
    Value::sptr physics_loss;
    Value::sptr ic_loss;
};
PhysicsResult pendulum(const MLP& net, double h, const Trident& t_end, double gamma, double angfreqsqr, double theta0, double omega0, int batch_size) {
    Value::sptr physics_loss = Value::create(0.0);
    double t_max = t_end.val->data;
    uniform_real_distribution<double> time_dist(0.0, t_max);
    for (int b = 0; b < batch_size; b++) {
        double time = time_dist(g_rng);
        auto tau = time / 10.0;
        Trident t(Value::create(tau), Value::create(1.0), Value::create(0.0));
        vector<Trident> t_vec{t};
        Trident theta = net(t_vec)[0];
        Value::sptr r = theta.de2 + 10 * gamma * theta.de1 + 100 * angfreqsqr * sin(theta.val);
        physics_loss  = physics_loss + r*r;
    }
    physics_loss = physics_loss / batch_size;
    Trident t0(Value::create(0.0), Value::create(1.0), Value::create(0.0));
    std::vector<Trident> t0_vec{t0};
    Trident ic_pred = net(t0_vec)[0];

    Value::sptr ic_loss = pow(ic_pred.val - theta0, 2.0) + pow(ic_pred.de1/10.0 - omega0, 2.0);

    return {physics_loss, ic_loss};
}

int main() {
    MLP net(1, {64, 48, 32, 16, 1});
    double M = 1.0, g = 9.81, r_len = 0.2;
    double theta0 = pi / 3.0;
    double omega0 = 0.0;
    double I = 1.0 / 3.0 * M * (2 * r_len) * (2 * r_len);
    double C = 0.2;
    double angfreqsqr = M * g * r_len / I;
    double gamma = C / I;
    int batch_size = 128;

    Trident t_end(Value::create(10.0), Value::create(0.0), Value::create(0.0));
    double learning_rate = 0.000000001;
    double loss_scaling  = 1000.0;
    double h             = 0.01;
    ofstream another_csv("loss.csv");
    another_csv<<"physics_loss,ic_loss"<<endl;
    for(int i = 0; i < 500; i++) {
        auto result = pendulum(net, h, t_end, gamma, angfreqsqr, theta0, omega0, batch_size);
        Value::sptr loss = result.physics_loss + loss_scaling * result.ic_loss;
        net.zero_grad();
        loss->backward();
        for(auto &p: net.parameters()) {
            p.val->data -= learning_rate * p.val->grad;
        }
        cout<<i<<" "<<"physics_loss = "<<result.physics_loss->data<<" ic_loss = "<<result.ic_loss->data<<endl;
        another_csv<<result.physics_loss->data<<","<<result.ic_loss->data<<endl;
    }
    //dtau = dt /10
    // dtheta / dt = dtheta / (dtau * 10)
    // d2theta / dt2 = d/dt dtheta /dt = d/10*dtau dtheta / 10*dtau
    // d2theta / dt2 = d2theta / (d2tau*100)

    ofstream csv("pendulum_results_NN.csv");
    csv<<"time, theta, omega"<<endl;
    int n_steps = 1000;
    double t_scale = 10.0;
    for (int i = 0; i <= n_steps; i++) {
        double time = t_end.val->data * i / n_steps;   // 0..10s
        double tau = time / t_scale;

        Trident t(Value::create(tau), Value::create(1.0), Value::create(0.0));
        vector<Trident> t_vec{t};
        Trident theta = net(t_vec)[0];

        double theta_val = theta.val->data;
        double omega_val = theta.de1->data / t_scale;  // dtheta/dt, undo the tau scaling

        csv << time << "," << theta_val << "," << omega_val << "\n";
    }
    csv.close();
    return 0;
}
