#pragma once
#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

#include <vector>
#include <cmath>
#include <algorithm>

using ActivationFunc = float(*)(float);

struct ActivationFunction {
    ActivationFunc activate;
    ActivationFunc derive;

    ActivationFunction(ActivationFunc a = nullptr, ActivationFunc d = nullptr)
        : activate(a), derive(d) {}
};

namespace Activations {
    // Funciones de activación
    inline float sigmoid(float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }

    inline float sigmoid_derivative(float y) {
        return y * (1.0f - y);
    }

    inline float relu(float x) {
        return x > 0 ? x : 0.0f;
    }

    inline float relu_derivative(float x) {
        return x > 0 ? 1.0f : 0.0f;
    }

    inline float tanh_act(float x) {
        return std::tanh(x);
    }

    inline float tanh_derivative(float y) {
        return 1.0f - y * y;
    }

    inline std::vector<float> softmax(const std::vector<float>& z) {
        std::vector<float> res(z.size());
        float max_z = *std::max_element(z.begin(), z.end());
        float sum = 0.0f;

        for (size_t i = 0; i < z.size(); ++i) {
            res[i] = exp(z[i] - max_z);
            sum += res[i];
        }

        for (auto& val : res) {
            val /= sum;
        }

        return res;
    }

    inline ActivationFunction Sigmoid() {
        return ActivationFunction(sigmoid, sigmoid_derivative);
    }

    inline ActivationFunction ReLU() {
        return ActivationFunction(relu, relu_derivative);
    }

    inline ActivationFunction Tanh() {
        return ActivationFunction(tanh_act, tanh_derivative);
    }

    inline ActivationFunction Softmax() {
        return ActivationFunction(nullptr, nullptr);
    }
}

static inline int getActivationId(const ActivationFunction& act) {
        using namespace Activations;
        if (act.activate == Sigmoid().activate && act.derive == Sigmoid().derive) return 0;
        if (act.activate == ReLU().activate && act.derive == ReLU().derive) return 1;
        if (act.activate == Tanh().activate && act.derive == Tanh().derive) return 2;
        if (act.activate == Softmax().activate && act.derive == Softmax().derive) return 3;
        return -1;
    }

    static inline ActivationFunction getActivationById(int id) {
        using namespace Activations;
        switch (id) {
            case 0: return Sigmoid();
            case 1: return ReLU();
            case 2: return Tanh();
            case 3: return Softmax();
            default: throw std::runtime_error("Unknown activation id in model loading");
        }
    }
#endif // ACTIVATIONS_HPP