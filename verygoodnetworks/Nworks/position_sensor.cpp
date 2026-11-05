
#include <iostream> // optional, for debugging output
#include "neural_network.h"
#include "position_sensor.h"
#include "organism.h"
// Constructor
PositionSensor::PositionSensor(NeuralNetwork* network, Organism* organism)
    : network{ network }, organism{ organism } {
    input = 2;
    output = 0;

    inputStartingNeuron = organism->inputsUsed;
    //std::cout << "Bad Sensor: " << inputStartingNeuron << '\n';

}
PositionSensor::PositionSensor() {
    input = 2;
    output = 0;
}


std::unique_ptr<Module> PositionSensor::Clone(Organism* newOrganism, NeuralNetwork* newNetwork) const {
    return std::make_unique<PositionSensor>(newNetwork, newOrganism);
}
// Update method
void PositionSensor::Update() {
    if (network == nullptr || organism == nullptr) {
        std::cerr << "PositionSensor not properly initialized!" << std::endl;
        return;
    }

    // std::cout << "module: " << inputStartingNeuron << '\n';
     // Update the network's neurons.

    network->neurons[0 + inputStartingNeuron].activation = organism->x/1300;
    network->neurons[1 + inputStartingNeuron].activation = organism->y/768;
 

    for (int i = 0; i < input; i++) {
        network->neurons[i + inputStartingNeuron].module = this;
    }
    //    std::cout << "Food Sensor: " << inputStartingNeuron << '\n';

}
