#pragma once
#ifndef ORGANISM_SENSOR_H
#define ORGANISM_SENSOR_H

#include "module.h"
#include <vector>
#include <memory>

class Organism;
class NeuralNetwork;

class OrganismSensor : public Module {
public:
    std::vector<std::unique_ptr<Organism>>* organismList;
    Organism* organism;
    NeuralNetwork* network;

    OrganismSensor();
    OrganismSensor(NeuralNetwork* network = nullptr, Organism* organism = nullptr, std::vector<std::unique_ptr<Organism>>* organismList = nullptr);

    virtual void Update() override;
    virtual std::unique_ptr<Module> Clone(Organism* newOrganism, NeuralNetwork* newNetwork) const override;

};

#endif /
