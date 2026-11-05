#include "teleport_module.h"
#include "neural_network.h"
#include <iostream>

TeleportModule::TeleportModule()
    : network(nullptr), organism(nullptr) {
    input = 0;
    output = 2;
}

TeleportModule::TeleportModule(NeuralNetwork* network, Organism* organism)
    : network(network), organism(organism) {
    input = 0;
    output = 2;
}

void TeleportModule::Update() {
    if (network == nullptr || organism == nullptr) {
        std::cerr << "MovementModule not properly initialized!" << std::endl;
        return;
    }

    int speed = 2;

    if (network->outputIndices.size() >= output) {
        organism->x = network->neurons[network->outputIndices[0]].activation * 1300;
        organism->y = network->neurons[network->outputIndices[1]].activation * 768;
      
        // Decay output activations
        
        for (int i = 0; i < output; ++i) {
            network->neurons[network->outputIndices[i]].module = this;
        }
    }
}

std::unique_ptr<Module> TeleportModule::Clone(Organism* newOrganism, NeuralNetwork* newNetwork) const {
    return std::make_unique<TeleportModule>(newNetwork, newOrganism);
}
