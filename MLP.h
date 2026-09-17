#ifndef MLP_H
#define MLP_H

#include "Autograd.h"
#include <random>
#include <vector>
#include <cassert>
using namespace std;

inline random_device rd;
inline mt19937 g_rng(rd());
inline uniform_real_distribution<double> udb(-1.0, 1.0);

class Neuron {
  public:
    vector<Trident> weight;
    Trident bias = Trident(Value::create(udb(g_rng)), Value::create(0.0), Value::create(0.0));
    bool nonlin;
    int nin;

    Neuron(bool nonlin = true, int nin = 0) : nonlin(nonlin), nin(nin) {
        weight.reserve(nin);
        for (int i = 0; i < nin; ++i) {
            weight.push_back(Trident(
                Value::create(udb(g_rng)),
                Value::create(0.0),
                Value::create(0.0)
            ));
        }
    }

    Trident operator()(const vector<Trident>& x) const {
        assert(static_cast<int>(x.size()) == nin && "Neuron input size mismatch");
        Trident sum = bias;
        for (int i = 0; i < nin; ++i) {
            sum = sum + (x[i] * weight[i]);
        }
        return nonlin ? sum.Tanh() : sum;
    }

    vector<Trident> parameters() const {
        vector<Trident> params;
        params.reserve(nin + 1);
        for (int i = 0; i < nin; ++i) {
            params.push_back(weight[i]);
        }
        params.push_back(bias);
        return params;
    }
};

class Layer {
  public:
    vector<Neuron> neurons;
    bool nonlin;
    int nin;
    int nout;

    Layer(bool nonlin = true, int nin = 0, int nout = 0) : nonlin(nonlin), nin(nin), nout(nout) {
        neurons.reserve(nout);
        for (int i = 0; i < nout; ++i) {
            neurons.push_back(Neuron(nonlin, nin));
        }
    }

    vector<Trident> operator()(const vector<Trident>& x) const {
        assert(static_cast<int>(x.size()) == nin && "Layer input size mismatch");
        vector<Trident> y;
        y.reserve(nout);
        for (int i = 0; i < nout; ++i) {
            y.push_back(neurons[i](x));
        }
        return y;
    }

    vector<Trident> parameters() const {
        vector<Trident> y;
        for (int i = 0; i < nout; ++i) {
            auto p = neurons[i].parameters();
            y.insert(y.end(), p.begin(), p.end());
        }
        return y;
    }
};

class MLP {
  public:
    vector<Layer> layers;
    int nin;
    vector<int> nouts;

    MLP(int nin = 0, vector<int> nouts = {}) : nin(nin), nouts(nouts) {
        vector<int> sizes = {nin};
        sizes.insert(sizes.end(), nouts.begin(), nouts.end());
        for (size_t i = 0; i < nouts.size(); ++i) {
            bool is_nonlin = (i != nouts.size() - 1);
            layers.emplace_back(is_nonlin, sizes[i], sizes[i + 1]);
        }
    }

    vector<Trident> operator()(const vector<Trident>& x_in) const {
        vector<Trident> x = x_in;
        for (const Layer& layer : layers) {
            x = layer(x);
        }
        return x;
    }

    vector<Trident> parameters() const {
        vector<Trident> y;
        for (const Layer& layer : layers) {
            auto p = layer.parameters();
            y.insert(y.end(), p.begin(), p.end());
        }
        return y;
    }

    void zero_grad() const {
        for (auto& p : parameters()) {
            p.val->grad = 0.0;
            p.de1->grad = 0.0;
            p.de2->grad = 0.0;
        }
    }
};
#endif
