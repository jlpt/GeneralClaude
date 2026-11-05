#pragma once

#include "module.h"
#include "slice.h"
#include <vector>
#include <memory>

class Organism;
class NeuralNetwork;

class SliceSensor : public Module {
public:
    std::vector<Slice>* sliceList;
    Organism* organism;
    NeuralNetwork* network;

    SliceSensor();
    SliceSensor(NeuralNetwork* network = nullptr, Organism* organism = nullptr, std::vector<Slice>* sliceList = nullptr);

    virtual void Update() override;
    virtual std::unique_ptr<Module> Clone(Organism* newOrganism, NeuralNetwork* newNetwork) const override;
  
};

