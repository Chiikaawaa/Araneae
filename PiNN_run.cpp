#include "Autograd.h"
#include "MLP.h"
#include "ModelSL.h"
#include "Parameter_dial.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

int main() {
    cout<<"program start"<<endl;
    Training_params trn;
    Pendulum_params pnd;
    double total_time = trn.total_oscillations * pnd.time_period;
    int ns = total_time / trn.h;
    MLP net(1, {32, 32, 32, 1});
    load_model(net, "pendulum");
    ofstream csv("pendulum_results_NN.csv");
    csv<<"time,theta,omega"<<endl;
    for (int i = 0; i < ns; i++) {
        if(i%10==0) cout<<i<<'\n';
        double time = total_time * i / ns;
        double tau  = time * sqrt(pnd.angfreqsqr);

        Trident t(Value::create(tau), Value::create(1.0), Value::create(0.0));
        vector<Trident> t_vec{t};
        Trident theta = net(t_vec)[0];

        double theta_val = theta.val->data;
        double omega_val = theta.de1->data * sqrt(pnd.angfreqsqr);

        csv << time << "," << theta_val << "," << omega_val << "\n";
    }
    csv.close();
    cout << "Wrote " << ns << " data rows\n";
    return 0;
}
