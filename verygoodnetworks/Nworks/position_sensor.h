#pragma once


#include "module.h"
#include <vector>
#include <memory>

class Organism;
class NeuralNetwork;

class PositionSensor : public Module {
public:
    
    Organism* organism;
    NeuralNetwork* network;
    int inputStartingNeuron = 0;
    int outputStartingNeuron = 0;

    PositionSensor();
    PositionSensor(NeuralNetwork* network = nullptr, Organism* organism = nullptr);

    virtual void Update() override;
    virtual std::unique_ptr<Module> Clone(Organism* newOrganism, NeuralNetwork* newNetwork) const override;

};

