#include "Autograd.h"
#include "MLP.h"
#include "ModelSL.h"
#include "Parameter_dial.h"

#include <cmath>
#include <random>
#include <vector>
#include <fstream>
#include <chrono>
#include <string>

using namespace std;

struct PhysicsResult {
    Value::sptr physics_loss;
    Value::sptr ic_loss;
};

PhysicsResult pendulum(const MLP& net, const Trident& t_end, const Pendulum_params& pnd, const Training_params& trn) {
    Value::sptr physics_loss = Value::create(0.0);
    double t_max = t_end.val->data;
    auto T = trn.total_oscillations * pnd.time_period;

    std::uniform_real_distribution<double> time_dist(0.0, t_max);
    for(int b = 0; b < trn.batch_size; b++) {
        double time = time_dist(g_rng);
        double tau  = time * sqrt(pnd.angfreqsqr);
        Trident t(Value::create(tau), Value::create(1.0), Value::create(0.0));
        Trident theta = net(encode(t))[0];

        Value::sptr r = theta.de2 + pnd.gamma * theta.de1 / sqrt(pnd.angfreqsqr) + sin(theta.val);
        physics_loss = physics_loss + r*r;
    }
    physics_loss = physics_loss / trn.batch_size;

    Trident t0(Value::create(0.0), Value::create(1.0), Value::create(0.0));
    Trident ic_pred = net(encode(t0))[0];
    Value::sptr ic_loss = pow(ic_pred.val - pnd.theta0, 2.0) + pow(ic_pred.de1 - (pnd.omega0 / sqrt(pnd.angfreqsqr)), 2.0);

    return {physics_loss, ic_loss};
}

int main() {

    MLP net(n_inputs, {32, 32, 32, 1});

    double total_time = pnd.time_period * trn.total_oscillations;
    Trident t_end(Value::create(total_time), Value::create(0.0), Value::create(0.0));

    auto params = net.parameters();
    vector<double> m(params.size(), 0.0);
    vector<double> v(params.size(), 0.0);

    std::vector<double> pylo(trn.iterations+10);
    std::vector<double> iclo(trn.iterations+10);

    auto t_start = std::chrono::steady_clock::now();
    for(int i = 1; i <= trn.iterations; i++) {
        auto result = pendulum(net, t_end, pnd, trn);
        Value::sptr loss = trn.lambda_ph * result.physics_loss + trn.lambda_ic * result.ic_loss;
        net.zero_grad();
        loss->backward();
        auto invbias2 = 1 / (1 - pow(trn.beta2, i));
        auto invbias1 = 1 / (1 - pow(trn.beta1, i));

        trn.learning_rate = trn.lr_start * pow(trn.lr_end / trn.lr_start, i * 1.0/trn.iterations);
        for(size_t j = 0; j < params.size(); j++){

            v[j] = trn.beta2 * v[j] + (1 - trn.beta2) * params[j]->grad * params[j]->grad;
            auto v_estimator = v[j] * invbias2;

            m[j] = trn.beta1 * m[j] + (1 - trn.beta1) * params[j]->grad;
            auto m_estimator = m[j] * invbias1;

            params[j]->data -= trn.learning_rate * m_estimator / (sqrt(v_estimator) + trn.epsilon);

        }
        if(i % 10 == 0) {
            double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now()-t_start).count();
            cout<<i<<" "<<"physics_loss = "<<result.physics_loss->data<<" ic_loss = "<<result.ic_loss->data<<" elapsed = "<<elapsed<<"\n";

        }
        pylo[i] = result.physics_loss->data;
        iclo[i] = result.ic_loss->data;
    }
    if(!save_model(net, "pendulum")){
        std::cerr<<"Saving model failed"<<"\n";
    }
    else{
        cout<<"Model saved successfully"<<"\n";
    }

    double total = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    cout << "Training took " << total << " s\n";

    ofstream another_csv("loss.csv");
    another_csv<<"iteration,physics_loss,ic_loss"<<"\n";
    for(int k = 1; k <= trn.iterations; k++){
        another_csv<<k<<","<<pylo[k]<<","<<iclo[k]<<"\n";
    }
    another_csv.close();

    return 0;
}
