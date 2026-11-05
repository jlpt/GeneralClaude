#include "movement_module.h"
#include "neural_network.h"
#include <iostream>

MovementModule::MovementModule()
    : network(nullptr), organism(nullptr) {
    input = 0;
    output = 4;
}

MovementModule::MovementModule(NeuralNetwork* network, Organism* organism)
    : network(network), organism(organism) {
    input = 0;
    output = 4;
}

void MovementModule::Update() {
    if (network == nullptr || organism == nullptr) {
        std::cerr << "MovementModule not properly initialized!" << std::endl;
        return;
    }

    int speed = 3;

    if (network->outputIndices.size() >= 4) {
        organism->x += network->neurons[network->outputIndices[0]].activation * speed;
        organism->x -= network->neurons[network->outputIndices[1]].activation * speed;
        organism->y += network->neurons[network->outputIndices[2]].activation * speed;
        organism->y -= network->neurons[network->outputIndices[3]].activation * speed;

        // Decay output activations
        for (int i = 0; i < 4; ++i) {
            network->neurons[network->outputIndices[i]].activation *= 0.98f;
        }
        for (int i = 0; i < output; ++i) {
            network->neurons[network->outputIndices[i]].module = this;
        }
    }
}

std::unique_ptr<Module> MovementModule::Clone(Organism* newOrganism, NeuralNetwork* newNetwork) const {
    return std::make_unique<MovementModule>(newNetwork, newOrganism);
}
