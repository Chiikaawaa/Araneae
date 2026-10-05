#ifndef MODEL_H
#define MODEL_H

#include <iostream>
#include <fstream>
#include <string>
#include "Autograd.h"
#include "MLP.h"

inline bool save_model (const MLP& net, const string& path) {
    auto params = net.parameters();
    ofstream f(path, std::ios::binary);
    if(!f) return false;
    uint64_t n = static_cast<uint64_t>(params.size());
    f.write(reinterpret_cast<const char*> (&n), sizeof(n));
    for(auto& p: params) {
        auto d = p->data;
        f.write(reinterpret_cast<const char*>(&d), sizeof(d));
    }
    return true;
}

inline bool load_model (MLP& net, const string& path) {
    auto params = net.parameters();
    ifstream f(path, std::ios::binary);
    if (!f) return false;
    uint64_t n;
    f.read(reinterpret_cast<char*>(&n), sizeof(n));
    if (n != params.size()) {
        std::cerr<<"Parameter count mismatch\n";
        return false;
    }
    for(auto& p: params) {
        double d;
        f.read(reinterpret_cast<char*> (&d), sizeof(d));
        p-> data = d;
    }
    return true;
}

#endif
