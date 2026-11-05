#include "slice_module.h"
#include "neural_network.h"
#include "slice.h"
#include <iostream>



SliceModule::SliceModule(NeuralNetwork* network, Organism* organism, std::vector<Slice>* slices)
    : network(network), organism(organism), slices{ slices } {
    input = 0;
    output = 4;
    

}

void SliceModule::Update() {
    if (network == nullptr || organism == nullptr || slices == nullptr) {
        std::cerr << "Slices not properly initialized!" << std::endl;
        return;
    }
    //if (cooldown <= 0) {
    if (network->neurons[network->neurons.size() - 4 - outputStartingNeuron].activation >= 0.001f && network->neurons[network->neurons.size() - 3 - outputStartingNeuron].activation >= 0.001f  && network->neurons[network->neurons.size() - 2 - outputStartingNeuron].activation >= 0.001f && network->neurons[network->neurons.size() - 1 - outputStartingNeuron].activation >= 0.001f && cooldown <= 0) {
        slices->push_back(Slice(((network->neurons[network->neurons.size() - 4 - outputStartingNeuron].activation - 0.5) * 600) + organism->x, ((network->neurons[network->neurons.size() - 3 - outputStartingNeuron].activation - 0.5) * 600) + organism->y, ((network->neurons[network->neurons.size() - 2 - outputStartingNeuron].activation - 0.5) * 600) + organism->x, ((network->neurons[network->neurons.size() - 1 - outputStartingNeuron].activation - 0.5) * 600) + organism->y));
        cooldown = cooldownPeriod;
    }
    cooldown--;
  

    
}

// Clone: create a new MovementModule with the offspring's network and organism pointers.
std::unique_ptr<Module> SliceModule::Clone(Organism* newOrganism, NeuralNetwork* newNetwork) const {
    return std::make_unique<SliceModule>(newNetwork, newOrganism, slices);
}
