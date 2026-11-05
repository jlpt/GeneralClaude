#pragma once

#include "module.h"
#include <vector>
#include <memory>

class Organism;
class NeuralNetwork;

class AirSensor : public Module {
public:

    Organism* organism;
    NeuralNetwork* network;

    AirSensor();
    AirSensor(NeuralNetwork* network = nullptr, Organism* organism = nullptr);

    virtual void Update() override;

    virtual std::unique_ptr<Module> Clone(Organism* newOrganism, NeuralNetwork* newNetwork) const override;
};

