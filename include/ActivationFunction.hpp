#pragma once

using ActivationFunc = float(*)(float);

struct ActivationFunction {
    ActivationFunc activar;
    ActivationFunc derivar;

    ActivationFunction(ActivationFunc a, ActivationFunc d)
        : activar(a), derivar(d) {}
};
