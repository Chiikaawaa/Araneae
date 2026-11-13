#ifndef MLP_H
#define MLP_H

#include "Autograd.h"
#include "Parameter_dial.h"

#include <random>
#include <vector>
#include <cassert>
#include <stdexcept>

inline std::uniform_real_distribution<double> udb(-0.5, 0.5);

enum class Activation { Tanh, linear };

class Neuron {
  public:
    std::vector<Value> weight;
    Value bias;
    bool nonlin;
    int nin;

    explicit Neuron(int nin, Activation act)
    : nonlin(act == Activation::Tanh), nin(nin) {
        if (nin <= 0) throw std::invalid_argument("Neuron: nin must be > 0");
        weight.reserve(nin);
        for (int i = 0; i < nin; ++i) {
            weight.push_back( Value::create( udb(g_rng) ) );
        }
        bias = Value::create(0.0);
    }

    Trident operator()(const std::vector<Trident>& x) const {
        assert(static_cast<int>(x.size()) == nin && "Neuron input size mismatch");
        Trident sum = x[0] * weight[0];
        for (int i = 1; i < nin; ++i) {
            sum = sum + (x[i] * weight[i]);
        }
        sum = sum + bias;
        return nonlin ? sum.Tanh() : sum;
    }

    std::vector<Value> parameters() const {
        std::vector<Value> params;
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
    std::vector<Neuron> neurons;
    bool nonlin;
    int nin;
    int nout;

    explicit Layer(int nin, int nout, Activation act)
    : nonlin(act == Activation::Tanh), nin(nin), nout(nout) {
        if (nin <= 0 || nout <= 0) throw std::invalid_argument("Layer: nin and nout must be > 0");
        neurons.reserve(nout);
        for (int i = 0; i < nout; ++i) {
            neurons.push_back(Neuron(nin, act));
        }
    }

    std::vector<Trident> operator()(const std::vector<Trident>& x) const {
        assert(static_cast<int>(x.size()) == nin && "Layer input size mismatch");
        std::vector<Trident> y;
        y.reserve(nout);
        for (int i = 0; i < nout; ++i) {
            y.push_back(neurons[i](x));
        }
        return y;
    }

    std::vector<Value> parameters() const {
        std::vector<Value> y;
        for (int i = 0; i < nout; ++i) {
            auto p = neurons[i].parameters();
            y.insert(y.end(), p.begin(), p.end());
        }
        return y;
    }
};

class MLP {
  public:
    std::vector<Layer> layers;
    int nin;
    std::vector<int> nouts;
    std::uint32_t first_id = 0;
    std::uint32_t n_params = 0;

    explicit MLP(int nin = 0, std::vector<int> nouts = {})
    : nin(nin), nouts(nouts) {
        if (nin <= 0 || nouts.empty()) throw std::invalid_argument("MLP: need nin > 0 and at least one layer");
        for (int n : nouts) {
            if (n <= 0) throw std::invalid_argument("MLP: layer widths must be > 0");
        }
        first_id = static_cast<std::uint32_t>(Tape.Nodes.size());
        std::vector<int> sizes = {nin};
        sizes.insert(sizes.end(), nouts.begin(), nouts.end());
        layers.reserve(nouts.size());
        for (size_t i = 0; i < nouts.size(); ++i) {
            Activation act = (i != nouts.size() - 1) ? Activation::Tanh : Activation::linear;
            layers.emplace_back(sizes[i], sizes[i + 1], act);
        }
        n_params = static_cast<std::uint32_t>(Tape.Nodes.size()) - first_id;

        std::uint32_t expected = 0;
        for(size_t i = 0; i < nouts.size(); i++) {
            expected += static_cast<std::uint32_t>(sizes[i+1] * (sizes[i] + 1));
        }
        assert(n_params == expected && "unexpected nodes pushed while building the MLP");
        (void)expected;

    }

    MLP(const MLP&) = delete;
    MLP& operator=(const MLP&) = delete;
    MLP(MLP&&) = default;
    MLP& operator=(MLP&&) = default;

    std::vector<Trident> operator()(const std::vector<Trident>& x_in) const {
        if(layers.empty()) return x_in;
        std::vector<Trident> x = layers[0](x_in);
        for(size_t i = 1; i < layers.size(); ++i) {
            x = layers[i](x);
        }
        return x;
    }

    std::vector<Value> parameters() const {
        std::vector<Value> y;
        y.reserve(n_params);
        for (const Layer& layer : layers) {
            auto p = layer.parameters();
            y.insert(y.end(), p.begin(), p.end());
        }
        return y;
    }

    void zero_grad() {
        assert(first_id + n_params <= Tape.mark && "call Tape.freeze() after building the model");
        for (std::uint32_t i = first_id; i < first_id + n_params; ++i) {
            Tape.Nodes[i].grad = 0.0;
        }
    }
};
#endif
