#include "Autograd.h"
#include "MLP.h"
#include "ModelSL.h"
#include "Parameter_dial.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

int main() {
    double total_time = (trn.total_oscillations) * pnd.time_period;
    int ns = total_time / trn.h;
    MLP net(n_inputs, {32, 32, 32, 1});
    if(!load_model(net, "modelmodel")) {
        std::cerr<<"Couldnt load model"<<"\n";
    }
    else {
        std::cout<<"Load kela model, thamb pudhcha karto"<<"\n";
    }
    Tape.freeze();
    std::ofstream csv("pendulum_results_NN.csv");
    csv<<"time,theta,omega,energy\n";
    for (int i = 0; i < ns; i++) {
        if(i%10==0) std::cout<<i<<'\n';
        double time = total_time * i / ns;
        double tau  = time * sqrt(pnd.angfreqsqr);

        Trident t(Value::create(tau), Value::create(1.0), Value::create(0.0));
        Trident theta = net(encode(t))[0];

        double theta_val = theta.val.data();
        double omega_val = theta.de1.data() * sqrt(pnd.angfreqsqr);

        double energy = 0.5 * pnd.I * omega_val * omega_val + pnd.M * pnd.g * pnd.r * (1 - cos(theta_val));

        csv << time << "," << theta_val << "," << omega_val << "," << energy << "\n";
        Tape.reset();
    }
    csv.close();
    std::cout << "Wrote " << ns << " data rows\n";
    return 0;
}
