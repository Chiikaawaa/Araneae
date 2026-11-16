//compile ->  g++ -std=c++23 -O2 -march=native -DNDEBUG pendulum_train.cpp -o pendulum_train

#include "Autograd.h"
#include "MLP.h"
#include "Adam.h"
#include "ModelSL.h"
#include "Parameter_dial.h"
#include "pendulum.h"

#include <fstream>
#include <chrono>
#include <omp.h>
#include <random>

int main(int argc, char* argv[]) {

    int nthreads = omp_get_max_threads();
    if(argc > 1) {
        nthreads = std::stoi(argv[1]);
    }
    if(argc > 2) {
        trn.iterations = std::stoi(argv[2]);
    }
    double total_time = pnd.time_period * trn.total_oscillations;
    std::uniform_real_distribution<double> time_dist(0.0, total_time);
    std::vector<double> times(trn.batch_size, 0.0);

    std::vector<double> pylo(trn.iterations, 0.0);
    std::vector<double> iclo(trn.iterations, 0.0);

    MLP net(n_inputs, {32,32,32,1});
    Tape.freeze();
    std::vector<Value> params = net.parameters();
    std::vector<basic> prefix(Tape.basicNodes.begin(), Tape.basicNodes.begin()+ Tape.bmark);
    Adam adam(net.n_params);

    std::vector<double> pdata(net.n_params);
    std::vector<std::vector<double>> gbuf(nthreads, std::vector<double>(net.n_params, 0.0));
    std::vector<double> g(net.n_params, 0.0);
    std::vector<double> pl(nthreads, 0.0);
    double icv = 0.0;
    int used = nthreads;

    auto t_start = std::chrono::steady_clock::now();
    for(int i = 0; i < trn.iterations; i++) {
        for(auto& x : times) {
            x = time_dist(g_rng);
        }
        for(int j = 0; j< net.n_params; j++) {
            pdata[j] = params[j].data();
        }

        #pragma omp parallel num_threads(nthreads)
        {
            int threadID = omp_get_thread_num();
            int obtainedThreads = omp_get_num_threads();
            used = obtainedThreads;
            if(!Tape.frozen){
                Tape.basicNodes.assign(prefix.begin(),prefix.end());
                Tape.freeze();
            }
            for(int j = 0; j < net.n_params; j++) {
                Tape.basicNodes[net.first_id + j].data = pdata[j];
                Tape.basicNodes[net.first_id + j].grad = 0.0;
            }
            int lo = threadID * trn.batch_size / obtainedThreads;
            int hi = (threadID + 1) * trn.batch_size / obtainedThreads;
            PhysicsResults result = pendulum(net, times, lo, hi, threadID == 0);
            Value loss = trn.lambda_ph * result.physics_loss + trn.lambda_ic * result.ic_loss;
            loss.backward();
            pl[threadID] = result.physics_loss.data();
            if(!threadID) icv = result.ic_loss.data();
            for(int j = 0; j < net.n_params; j++) {
                gbuf[threadID][j] = Tape.basicNodes[net.first_id + j].grad;
            }
            Tape.reset();
        }
        double physics = 0.0;
        for(int j = 0; j < net.n_params; j++) g[j] = 0.0;
        for(int t = 0; t < used; t++) {
            physics += pl[t];
            for(int j = 0; j < net.n_params; j++) g[j] += gbuf[t][j];
        }
        adam.step(params, g, i+1);
        if(i % 100 == 0 && i !=0) {
            double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now()-t_start).count();
            std::cout<<i<<" "<<"physics_loss = "<<physics<<" ic_loss = "<<icv<<" elapsed = "<<elapsed<<"\n";

        }
        pylo[i] = physics;
        iclo[i] = icv;
    }
    if(!save_model(net, "modelmodel")){
        std::cerr<<"Saving model failed"<<"\n";
    }
    else{
        std::cout<<"Model saved successfully"<<"\n";
    }

    double total = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    std::cout << "Training took " << total << " s\n";

    std::ofstream another_csv("loss.csv");
    another_csv<<"iteration,physics_loss,ic_loss"<<"\n";
    for(int k = 0; k < trn.iterations; k++){
        another_csv<<k<<","<<pylo[k]<<","<<iclo[k]<<"\n";
    }
    another_csv.close();

    return 0;
}
