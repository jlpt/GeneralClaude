#ifndef SENSOR_H
#define SENSOR_H

#include "module.h"
#include "food.h"
#include <vector>
#include <memory>

class Organism;
class NeuralNetwork;

class FoodSensor : public Module {
public:
    std::vector<Food>* foodList;
    Organism* organism;
    NeuralNetwork* network;
    int inputStartingNeuron = 0;
    int outputStartingNeuron = 0;

    FoodSensor();
    FoodSensor(NeuralNetwork* network = nullptr, Organism* organism = nullptr, std::vector<Food>* foodList = nullptr);

    virtual void Update() override;
    virtual std::unique_ptr<Module> Clone(Organism* newOrganism, NeuralNetwork* newNetwork) const override;

};

#endif // SENSOR_H
