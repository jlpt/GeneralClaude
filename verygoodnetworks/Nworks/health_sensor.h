#pragma once

#include "module.h"
#include <vector>
#include <memory>

class Organism;
class NeuralNetwork;

class HealthSensor : public Module {
public:
  
    Organism* organism;
    NeuralNetwork* network;

    HealthSensor();
    HealthSensor(NeuralNetwork* network = nullptr, Organism* organism = nullptr);

    virtual void Update() override;
  
    virtual std::unique_ptr<Module> Clone(Organism* newOrganism, NeuralNetwork* newNetwork) const override;
};

