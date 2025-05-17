#pragma once
#include <cmath>
#include <ActivationFunction.hpp>

namespace Activations {
    static float sigmoid(float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }
    static float sigmoid_derivative(float y) {
        return y * (1.0f - y);
    }

    static float relu(float x) {
        return x > 0 ? x : 0.0f;
    }
    static float relu_derivative(float x) {
        return x > 0 ? 1.0f : 0.0f;
    }

    static float tanh_act(float x) {
        return std::tanh(x);
    }
    static float tanh_derivative(float y) {
        return 1.0f - y * y;
    }

    static ActivationFunction Sigmoid() {
        return ActivationFunction(sigmoid, sigmoid_derivative);
    }

    static ActivationFunction ReLU() {
        return ActivationFunction(relu, relu_derivative);
    }

    static ActivationFunction Tanh() {
        return ActivationFunction(tanh_act, tanh_derivative);
    }
}
