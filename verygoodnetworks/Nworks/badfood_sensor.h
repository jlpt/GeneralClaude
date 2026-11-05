#pragma once


#include "module.h"
#include "bad_food.h"
#include <vector>
#include <memory>

class Organism;
class NeuralNetwork;

class BadFoodSensor : public Module {
public:
    std::vector<BadFood>* foodList;
    Organism* organism;
    NeuralNetwork* network;
    int inputStartingNeuron = 0;
    int outputStartingNeuron = 0;

    BadFoodSensor();
    BadFoodSensor(NeuralNetwork* network = nullptr, Organism* organism = nullptr, std::vector<BadFood>* foodList = nullptr);

    virtual void Update() override;
    virtual std::unique_ptr<Module> Clone(Organism* newOrganism, NeuralNetwork* newNetwork) const override;

};


