#include<iostream>
#include<vector>
#include<cmath>
#define pi 3.14159265358979323846
using namespace std;

double M = 1.0, g = 9.81, r = 0.2, θ0 = pi/3;
double ω0 = 0.0;
double I = 1.0 / 3.0 * M * (2 * r) * (2 * r);
double C = 0.2;


vector<double> func(vector<double> Y){
    double θ  = Y[0];
    double ω  = Y[1];
    double dθ = ω;
    double dω = ((-1 * C * ω) - (M * g * r * sin(θ)))/I;
    return {dθ, dω};
}

vector<double> rk4(vector<double> Y, double h){
    vector<double> z = func(Y);
    vector<double> k1;
    for(auto x : z){
        k1.emplace_back(x * h);
    }
    vector<double> Y2;
    for(int i = 0; i < Y.size(); i++){
        Y2.emplace_back(Y[i] + k1[i]/2);
    }
    z = func(Y2);
    vector<double> k2;
    for(auto x : z){
        k2.emplace_back(x * h);
    }
    vector<double> Y3;
    for(int i = 0; i < Y.size(); i++){
        Y3.emplace_back(Y[i] + k2[i]/2);
    }
    z = func(Y3);
    vector<double> k3;
    for(auto x : z){
        k3.emplace_back(x * h);
    }
    vector<double> Y4;
    for(int i = 0; i < Y.size(); i++){
        Y4.emplace_back(Y[i] + k3[i]);
    }
    z = func(Y4);
    vector<double> k4;
    for(auto x : z){
        k4.emplace_back(x * h);
    }
    vector<double> K;
    for(int i = 0; i < Y.size(); i++){
        K.emplace_back((k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])/6);
    }
    vector<double> Ynext;
    for(int i = 0; i < Y.size(); i++){
        Ynext.emplace_back(Y[i] + K[i]);
    }
    return Ynext;
}
double energy(double θ, double ω){
    return 0.5 * I * ω * ω + M * g * r * (1 - cos(θ));
}
int main() {
    vector<double> Y = {θ0, ω0};
        double h = 0.01;
        double t_end = 10.0;
        double t = 0.0;
        int n_steps = int(t_end / h);

        ofstream csv("rk4_results.csv");
        csv << "time,theta,omega\n";
        csv << t << "," << Y[0] << "," << Y[1] << "\n";   // t=0 initial condition

        for(int i = 0; i < n_steps; i++){
            Y = rk4(Y, h);
            t += h;
            csv << t << "," << Y[0] << "," << Y[1] << "\n";
        }
        csv.close();

        return 0;
}
