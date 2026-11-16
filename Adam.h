#ifndef ADAM_H
#define ADAM_H

#include "Autograd.h"
#include "Parameter_dial.h"

struct Adam {
    std::vector<double> m,v;

    explicit Adam(std::size_t n) :
    m(n, 0.0), v(n, 0.0) {}

    void step(const std::vector<Value>& params, const std::vector<double>& g, int i) {
        assert(params.size() == m.size() && g.size() == m.size());
        double invbias1 = 1.0 / (1.0 - std::pow(dlo.beta1, i));
        double invbias2 = 1.0 / (1.0 - std::pow(dlo.beta2, i));

        dlo.learning_rate = dlo.lr_start * std::pow(dlo.lr_end / dlo.lr_start, double(i) / trn.iterations);

        for (std::size_t j = 0; j < params.size(); j++) {
            v[j] = dlo.beta2 * v[j] + (1.0 - dlo.beta2) * g[j] * g[j];
            double v_estimator = v[j] * invbias2;

            m[j] = dlo.beta1 * m[j] + (1.0 - dlo.beta1) * g[j];
            double m_estimator = m[j] * invbias1;

            params[j].data() -= dlo.learning_rate * m_estimator / (std::sqrt(v_estimator) + dlo.epsilon);
        }
    }
};

#endif
