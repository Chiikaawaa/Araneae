#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include <cstdint>
#include <iostream>
#include <cmath>
#include <vector>
#include <memory>
#include <utility>
#include <string>
#include <array>
#include <limits>
#include <cassert>

constexpr std::uint32_t None = std::numeric_limits<std::uint32_t>::max();

struct node {
    double data;
    double grad;
    std::uint32_t p0;
    double d0;
    std::uint32_t p1;
    double d1;
};

struct tape {
    std::vector<node> Nodes;
    bool frozen = false;
    std::uint32_t mark = 0;
    tape() {
        Nodes.reserve( 2 << 20 );
    }
    std::uint32_t push(
        double data,
        std::uint32_t p0 = None,
        double d0 = 0,
        std::uint32_t p1 = None,
        double d1 = 0
        ) {
            Nodes.push_back({data, 0.0, p0, d0, p1, d1});
            return (std::uint32_t)Nodes.size() - 1;
        }
    void freeze() {
        mark = (std::uint32_t)Nodes.size();
        frozen = true;
    }
    void reset() {
        assert(frozen && "Tape.freeze() after building the model");
        Nodes.resize(mark);
    }
};
inline tape Tape;

class Value {
    public:
        std::uint32_t id = None;
        double& data() const {
            return Tape.Nodes[id].data;
        }
        double& grad() const {
            return Tape.Nodes[id].grad;
        }

        //future implementation
        /*enum Op{
            add,
            sub,
            mul,
            div,
            pow,
            exp,
            sin,
            cos,
            tanh
            };*/

    private:

    public:

        static Value create( double data ) {
            return Value{Tape.push(data)};
        }

        friend std::ostream& operator<<(std::ostream& os, const Value& v){
            return os << "Value(data = "<<v.data()<<", grad = "<<v.grad()<<" )";
        }

        friend Value operator + (const Value& a, const Value& b) {
            double out_data = a.data() + b.data();
            double da = 1.0;
            double db = 1.0;
            std::uint32_t id = Tape.push(out_data, a.id, da, b.id, db);
            return Value{id};
        }

        friend Value operator + (const Value& a, double k) {
            double out_data = a.data() + k;
            double da = 1.0;
            std::uint32_t id = Tape.push(out_data, a.id, da);
            return Value{id};
        }

        friend Value operator + (double k, const Value& a) {
            return a + k;
        }

        friend Value operator - (const Value& a, const Value& b) {
            double out_data = a.data() - b.data();
            double da = 1.0;
            double db = -1.0;
            std::uint32_t id = Tape.push(out_data, a.id, da, b.id, db);
            return Value{id};
        }

        friend Value operator - (const Value& a, double k) {
            double out_data = a.data() - k;
            double da = 1.0;
            std::uint32_t id = Tape.push(out_data, a.id, da);
            return Value{id};
        }

        friend Value operator - (double k, const Value& a) {
            double out_data = k - a.data();
            double da = -1.0;
            std::uint32_t id = Tape.push(out_data, a.id, da);
            return Value{id};
        }

        friend Value operator * (const Value& a, const Value& b) {
            double out_data = a.data() * b.data();
            double da = 1.0 * b.data();
            double db = a.data() * 1.0;
            std::uint32_t id = Tape.push(out_data, a.id, da, b.id, db);
            return Value{id};
        }

        friend Value operator * (const Value& a, double k) {
            double out_data = a.data() * k;
            double da = 1.0 * k;
            std::uint32_t id = Tape.push(out_data, a.id, da);
            return Value{id};
        }

        friend Value operator * (double k, const Value& a) {
            return a * k;
        }

        friend Value operator / (const Value& a, const Value& b) {
            double out_data = a.data() / b.data();
            double da = 1.0 / b.data();
            double db = a.data() * -1.0 / (b.data() * b.data());
            std::uint32_t id = Tape.push(out_data, a.id, da, b.id, db);
            return Value{id};
        }

        friend Value operator / (const Value& a, double k) {
            double out_data = a.data() / k;
            double da = 1.0 / k;
            std::uint32_t id = Tape.push(out_data, a.id, da);
            return Value{id};
        }

        friend Value operator / (double k, const Value& a) {
            double out_data = k / a.data();
            double da = k * -1.0 / (a.data() * a.data());
            std::uint32_t id = Tape.push(out_data, a.id, da);
            return Value{id};
        }

        friend Value exp(const Value& a) {
            double out_data = std::exp(a.data());
            double da = 1.0 * std::exp(a.data());
            std::uint32_t id = Tape.push(out_data, a.id, da);
            return Value{id};
        }

        friend Value tanh(const Value& a) {
            double out_data = std::tanh(a.data());
            double da = 1 - (out_data * out_data);
            std::uint32_t id = Tape.push(out_data, a.id, da);
            return Value{id};
        }

        friend Value pow(const Value& a, double k) {
            double out_data = std::pow(a.data(), k);
            double da = k * std::pow(a.data(), k-1);
            std::uint32_t id = Tape.push(out_data, a.id, da);
            return Value{id};
        }

        friend Value sin(const Value& a) {
            double s = std::sin(a.data());
            double c = std::cos(a.data());
            std::uint32_t id = Tape.push(s, a.id, c);
            return Value{id};
        }

        friend Value cos(const Value& a) {
            double s = std::sin(a.data());
            double c = std::cos(a.data());
            std::uint32_t id = Tape.push(c, a.id,-1.0 * s);
            return Value{id};
        }

        void backward() const {
            Tape.Nodes[id].grad = 1.0;
            for (std::uint32_t i = id + 1; i > Tape.mark; i--) {
                const node n = Tape.Nodes[i-1];
                if(n.grad == 0) continue;
                if (n.p0 != None) Tape.Nodes[n.p0].grad += n.d0 * n.grad;
                if (n.p1 != None) Tape.Nodes[n.p1].grad += n.d1 * n.grad;
            }
        }

};

class Trident {
    public:
        Value val,de1,de2;
        Trident(Value val, Value de1, Value de2):
        val(val),
        de1(de1),
        de2(de2)
        {}
    private:
        friend std::ostream& operator <<(std::ostream& os, const Trident& t){
            return os <<"Trident( val = " << t.val << ", de1 = "<< t.de1 << ", de2 = " << t.de2<<" )"<<"\n";
        }

    public:
        Trident operator + (const Trident& b) const {
            Value val = this->val + b.val;
            Value de1 = this->de1 + b.de1;
            Value de2 = this->de2 + b.de2 ;
            Trident out(val, de1, de2);
            return out;
        }
        Trident operator + (const Value& b) const {
            Value val = this->val + b;
            Value de1 = this->de1;
            Value de2 = this->de2;
            Trident out(val, de1, de2);
            return out;
        }
        Trident operator + (const double k) const {
            Value val = this-> val + k;
            Trident out(val, this->de1, this-> de2);
            return out;
        }
        friend Trident operator + (double k, const Trident& a) {
            Value val = a.val + k;
            Trident out(val, a.de1, a.de2);
            return out;
        }
        friend Trident operator + (const Value& v, const Trident& a) {
            Value val = a.val + v;
            Value de1 = a.de1;
            Value de2 = a.de2;
            Trident out(val, de1, de2);
            return out;
        }
        Trident operator - (double k) const {
            Value val = this-> val - k;
            Trident out(val, this->de1, this-> de2);
            return out;
        }
        Trident operator - (const Trident& b) const {
            Value val = this->val - b.val;
            Value de1 = this->de1 - b.de1;
            Value de2 = this->de2 - b.de2 ;
            Trident out(val, de1, de2);
            return out;
        }
        Trident operator - (const Value& b) const {
            Value val = this->val - b;
            Value de1 = this->de1;
            Value de2 = this->de2;
            Trident out(val, de1, de2);
            return out;
        }
        friend Trident operator - (double k, const Trident& a) {
            Value val = k - a.val;
            Trident out(val, -1*a.de1, -1*a.de2);
            return out;
        }
        friend Trident operator - (const Value& v, const Trident& a) {
            Value val = v - a.val;
            Value de1 = -1 * a.de1;
            Value de2 = -1 * a.de2;
            Trident out(val, de1, de2);
            return out;
        }
        Trident operator * (const Trident& b) const {
            Value val = this-> val * b.val;
            Value de1 = this-> de1 * b.val + this-> val * b.de1;
            Value de2 = this-> de2 * b.val + this-> de1 * b.de1 * 2 + b.de2 * this-> val;
            Trident out(val, de1, de2);
            return out;
        }
        Trident operator * (const Value& w) const {
            Value val = this->val * w;
            Value de1 = this->de1 * w;
            Value de2 = this->de2 * w;
            Trident out(val, de1, de2);
            return out;
        }
        Trident operator * (double k) const {
            Value val = this-> val * k;
            Value de1 = this-> de1 * k;
            Value de2 = this-> de2 * k;
            Trident out(val, de1, de2);
            return out;
        }
        friend Trident operator * (double k, const Trident& a) {
            return a * k;
        }
        friend Trident operator * (const Value& v, const Trident& a) {
            return a * v;
        }
        Trident operator / (const Trident& b) const {

            Value val = this-> val / b.val;
            Value de1 = (b.val * this-> de1 + this-> val * b.de1 * -1) / (b.val * b.val);
            Value de2 = (b.val * b.val * this->de2
                    - this->val * b.de2 * b.val
                    - 2.0 * this->de1 * b.de1 * b.val
                    + 2.0 * this->val * b.de1 * b.de1)
                    / (b.val * b.val * b.val);
            Trident out(val, de1, de2);
            return out;
        }
        Trident operator / (double k) const {
            Value val = this-> val / k;
            Value de1 = this-> de1 / k;
            Value de2 = this-> de2 / k;
            Trident out(val, de1, de2);
            return out;
        }
        friend Trident operator / (double k, const Trident& a) {
            Value val = k / a.val;
            Value de1 = (-1.0 * val) * a.de1 / a.val;
            Value de2 = (-2.0 * de1 * a.de1 - val * a.de2) / a.val;
            Trident out(val, de1, de2);
            return out;
        }
        Trident Exp() const {
            Value e   = exp(this->val);
            Value val = e;
            Value de1 = val * this->de1;
            Value de2 = e * this->de1 * this->de1 + e * this->de2;
            Trident out(val, de1, de2);
            return out;
        }
        Trident Tanh() const {
            Value t   = tanh(this->val);
            Value one_minust2 = (1 - t * t);
            Value val = t;
            Value de1 = one_minust2 * this->de1;
            Value de2 = -2 * t * one_minust2 * this->de1 * this->de1 + one_minust2 * this->de2;
            Trident out(val, de1, de2);
            return out;
        }
        Trident Pow(const double k) const {
            Value val = pow(this->val, k);
            Value de1 = pow(this->val, k-1) * k * this->de1;
            Value de2 = (pow(this->val, k-2) * this->de1 * this->de1 * (k-1) * k) + (pow(this->val, k-1) * this->de2 * k);
            Trident out(val, de1, de2);
            return out;
        }
        Trident Sin() const {
            Value s   = sin(this->val);
            Value c   = cos(this->val);
            Value val = s;
            Value de1 = c * this->de1;
            Value de2 = s * this->de1 * this->de1 * -1 + c * this->de2;
            Trident out(val, de1, de2);
            return out;
        }
        Trident Cos() const {
            Value s   = sin(this->val);
            Value c   = cos(this->val);
            Value val = c;
            Value de1 = s * -1 * this->de1;
            Value de2 = c * this->de1 * this->de1 * -1 + s * this->de2 * -1;
            Trident out(val, de1, de2);
            return out;
        }
};
#endif
