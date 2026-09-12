        #include<iostream>
        #include<cmath>
        #include<vector>
        #include<functional>
        #include<memory>
        #include<utility>
        #include<string>
        #include<unordered_set>

        using namespace std;

        class Value {
            public:
                using sptr = std::shared_ptr<Value>;
                double data;
                double grad;
            private:
                std::string _op;
                std::string label;
                std::vector<sptr> _prev;
                std::function<void()> _backward;

                Value(
                    double data = 0.0,
                    std::vector<sptr> _prev = {},
                    std::string _op = "",
                    std::string label = ""
                ):
                data(data),
                grad(0.0),
                _backward([]() {}),
                _prev(_prev),
                _op(_op),
                label(label)
                {}
                //Value(const Value&) = delete;
                //Value& operator = (const Value&) = delete;

            public:

                static sptr create(double data, std::vector<sptr> _prev = {}, std::string _op = "", std::string label = "") {
                    return sptr(new Value(data, std::move(_prev), std::move(_op), std::move(label)));
                }

                friend std::ostream& operator<<(std::ostream& os, const Value& v){
                    return os << "Value(data = "<<v.data<<", grad = "<<v.grad<<" )";
                }

                friend sptr operator + (const sptr& a, const sptr& b) {
                    auto out = Value::create(a->data + b->data, {a, b}, "+");
                    Value* out_rawptr = out.get();
                    out->_backward = [a, b, out_rawptr]() {
                        a->grad += 1.0 * out_rawptr->grad;
                        b->grad += 1.0 * out_rawptr->grad;
                    };
                    return out;
                }
                friend sptr operator + (const sptr& a, double k) {
                    auto out = Value::create(a->data + k, {a}, "scalar +");
                    Value* out_rawptr = out.get();
                    out->_backward = [a, out_rawptr]() {
                        a->grad += 1.0 * out_rawptr->grad;
                    };
                    return out;
                }
                friend sptr operator + (double k, const sptr& a) {
                    auto out = Value::create(a->data + k, {a}, "scalar +");
                    Value* out_rawptr = out.get();
                    out->_backward = [a, out_rawptr]() {
                        a->grad += 1.0 * out_rawptr->grad;
                    };
                    return out;
                }

                friend sptr operator - (const sptr& a, const sptr& b) {
                    auto out = Value::create(a->data - b->data, {a, b}, "-");
                    Value* out_rawptr = out.get();
                    out->_backward = [a, b, out_rawptr]() {
                        a->grad += 1.0 * out_rawptr->grad;
                        b->grad += -1.0 * out_rawptr->grad;
                    };
                    return out;
                }
                friend sptr operator - (const sptr& a, double k) {
                    auto out = Value::create(a->data - k, {a}, "scalar -");
                    Value* out_rawptr = out.get();
                    out->_backward = [a, out_rawptr]() {
                        a->grad += 1.0 * out_rawptr->grad;
                    };
                    return out;
                }
                friend sptr operator - (double k, const sptr& a) {
                    auto out = Value::create(k - a->data, {a}, "scalar -");
                    Value* out_rawptr = out.get();
                    out->_backward = [a, out_rawptr]() {
                        a->grad += -1.0 * out_rawptr->grad;
                    };
                    return out;
                }

                friend sptr operator * (const sptr& a, const sptr& b) {
                    auto out = Value::create(a->data * b->data, {a, b}, "*");
                    Value* out_rawptr = out.get();
                    out->_backward = [a, b, out_rawptr]() {
                        a->grad += b->data * out_rawptr->grad;
                        b->grad += a->data * out_rawptr->grad;
                    };
                    return out;
                }
                friend sptr operator * (const sptr& a, double k) {
                    auto out = Value::create(a->data * k, {a}, "scalar *");
                    Value* out_rawptr = out.get();
                    out->_backward = [a, k, out_rawptr]() {
                        a->grad += k * out_rawptr->grad;
                    };
                    return out;
                }
                friend sptr operator * (double k, const sptr& a) {
                    auto out = Value::create(a->data * k, {a}, "scalar *");
                    Value* out_rawptr = out.get();
                    out->_backward = [a, k, out_rawptr]() {
                        a->grad += k * out_rawptr->grad;
                    };
                    return out;
                }

                friend sptr operator / (const sptr& a, const sptr& b) {
                    auto out = Value::create(a->data / b->data, {a, b}, "div");
                    Value* out_rawptr = out.get();
                    out->_backward = [a, b, out_rawptr]() {
                        a->grad += 1 / b->data * out_rawptr->grad;
                        b->grad += (-a->data / (b->data * b->data)) * out_rawptr->grad;
                    };
                    return out;
                }
                friend sptr operator / (const sptr& a, double k) {
                    auto out = Value::create(a->data / k, {a}, "scalar div");
                    Value* out_rawptr = out.get();
                    out->_backward = [a, k, out_rawptr]() {
                        a->grad += 1/k * out_rawptr->grad;
                    };
                    return out;
                }
                friend sptr operator / (double k, const sptr& a) {
                    auto out = Value::create(k / a->data, {a}, "scalar div");
                    Value* out_rawptr = out.get();
                    out->_backward = [a, k, out_rawptr]() {
                        a->grad += -k * out_rawptr->grad / (a->data * a->data);
                    };
                    return out;
                }
                friend sptr exp(const sptr& a) {
                    auto out = Value::create(std::exp(a->data), {a}, "exp()");
                    Value* out_rawptr = out.get();
                    out->_backward = [a, out_rawptr]() {
                        a->grad += out_rawptr->data * out_rawptr->grad;
                    };
                    return out;
                }

                friend sptr tanh(const sptr& a) {
                    auto out = Value::create(std::tanh(a->data), {a}, "tanh()");
                    Value* out_rawptr = out.get();
                    out->_backward = [a, out_rawptr]() {
                        a->grad += (1.0 - out_rawptr->data * out_rawptr->data) * out_rawptr->grad;
                    };
                    return out;
                }

                friend sptr pow(const sptr& a, double k) {
                    auto out = Value::create(std::pow(a->data, k), {a}, "pow()");
                    Value* out_rawptr = out.get();
                    out->_backward = [a, k, out_rawptr]() {
                        a->grad += k * std::pow(a->data, k - 1) * out_rawptr->grad;
                    };
                    return out;
                }

                friend sptr sin(const sptr& a) {
                    double s = std::sin(a->data);
                    double c = std::cos(a->data);
                    auto out = Value::create(s, {a}, "sin()");
                    Value* out_rawptr = out.get();
                    out->_backward = [a, out_rawptr, c]() {
                        a->grad += c * out_rawptr->grad;
                    };
                    return out;
                }

                friend sptr cos(const sptr& a) {
                    double s = std::sin(a->data);
                    double c = std::cos(a->data);
                    auto out = Value::create(c, {a}, "cos()");
                    Value* out_rawptr = out.get();
                    out->_backward = [a, out_rawptr, s]() {
                        a->grad += -s * out_rawptr->grad;
                    };
                    return out;
                }

            void backward() {
                    std::vector<Value*> topo;
                    std::unordered_set<Value*> visited;

                    std::function<void(Value*)> build_topo = [&](Value* v) {
                        if (visited.insert(v).second) {
                            for (auto& child : v->_prev) build_topo(child.get());
                            topo.push_back(v);
                        }
                    };

                    build_topo(this);
                    this->grad = 1.0;
                    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
                        (*it)->_backward();
                    }
                }
                void zero_grad() {
                    std::unordered_set<Value*> visited;
                    std::function<void(Value*)> gd0 = [&](Value* v) {
                        if (visited.insert(v).second) {
                            for (auto& child : v->_prev) gd0(child.get());
                            v->grad = 0.0;
                        }
                    };
                    gd0(this);
                }
        };

        class Trident {
            public:
                Value::sptr val,de1,de2;
                Trident(Value::sptr val, Value::sptr de1, Value::sptr de2):
                val(val),
                de1(de1),
                de2(de2)
                {}
            private:
                friend std::ostream& operator <<(std::ostream& os, const Trident& t){
                    return os <<"Trident( val = " << *t.val << ", de1 = "<< *t.de1 << ", de2 = " << *t.de2<<" )"<<endl;
                }

            public:
            Trident operator + (const Trident& b){
                Value::sptr val = this->val + b.val;
                Value::sptr de1 = this->de1 + b.de1;
                Value::sptr de2 = this->de2 + b.de2 ;
                Trident out(val, de1, de2);
                return out;
            }
            Trident operator + (const double k){
                Value::sptr val = this-> val + k;
                Trident out(val, this->de1, this-> de2);
                return out;
            }
            Trident operator * (const Trident& b){
                Value::sptr val = this-> val * b.val;
                Value::sptr de1 = this-> de1 * b.val + this-> val * b.de1;
                Value::sptr de2 = this-> de2 * b.val + this-> de1 * b.de1 * 2 + b.de2 * this-> val;
                Trident out(val, de1, de2);
                return out;
            }
            Trident operator * (const double k){
                Value::sptr val = this-> val * k;
                Value::sptr de1 = this-> de1 * k;
                Value::sptr de2 = this-> de2 * k;
                Trident out(val, de1, de2);
                return out;
            }
            Trident operator / (const Trident& b){

                Value::sptr val = this-> val / b.val;
                Value::sptr de1 = (b.val * this-> de1 + this-> val * b.de1 * -1) / (b.val * b.val);
                Value::sptr de2 = (b.val * b.val * this->de2
                        - this->val * b.de2 * b.val
                        - 2.0 * this->de1 * b.de1 * b.val
                        + 2.0 * this->val * b.de1 * b.de1)
                        / (b.val * b.val * b.val);
                Trident out(val, de1, de2);
                return out;
            }
            Trident operator / (double k){
                Value::sptr val = this-> val / k;
                Value::sptr de1 = this-> de1 / k;
                Value::sptr de2 = this-> de2 / k;
                Trident out(val, de1, de2);
                return out;
            }
            Trident Exp(){
                Value::sptr val = exp(this-> val);
                Value::sptr de1 = val * this-> de1;
                Value::sptr de2 = exp(this->val) * this->de1 * this-> de1 + exp(this->val) * this-> de2;
                Trident out(val, de1, de2);
                return out;
            }
            Trident Tanh(){
                Value::sptr val = tanh(this->val);
                Value::sptr de1 = (tanh(this->val) * tanh(this->val) * -1 + 1) * this->de1;
                Value::sptr de2 = (tanh(this->val) * -2 * (tanh(this->val) * tanh(this->val) * -1 + 1)) * this->de1 * this->de1 + ((tanh(this->val) * tanh(this->val) * -1 + 1) * this->de2);
                Trident out(val, de1, de2);
                return out;
            }
            Trident Pow(const double k){
                Value::sptr val = pow(this->val, k);
                Value::sptr de1 = pow(this->val, k-1) * k * this->de1;
                Value::sptr de2 = (pow(this->val, k-2) * this->de1 * this->de1 * (k-1) * k) + (pow(this->val, k-1) * this->de2 * k);
                Trident out(val, de1, de2);
                return out;
            }
            Trident Sin(){
                Value::sptr val = sin(this->val);
                Value::sptr de1 = cos(this->val) * this->de1;
                Value::sptr de2 = sin(this->val) * this->de1 * this->de1 * -1 + cos(this->val) * this->de2;
                Trident out(val, de1, de2);
                return out;
            }
            Trident Cos(){
                Value::sptr val = cos(this->val);
                Value::sptr de1 = sin(this->val) * -1 * this->de1;
                Value::sptr de2 = cos(this->val) * this->de1 * this->de1 * -1 + sin(this->val) * this->de2 * -1;
                Trident out(val, de1, de2);
                return out;
            }

        };

        int main(){
            cout<<"compiles!!"<<endl;
            return 0;
        }
