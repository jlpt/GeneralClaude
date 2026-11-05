
#include <iostream> // optional, for debugging output
#include "neural_network.h"
#include "air_sensor.h"
#include "organism.h"
// Constructor
AirSensor::AirSensor(NeuralNetwork* network, Organism* organism)
    : network{ network }, organism{ organism } {
    input = 1;
    output = 0;
    inputStartingNeuron = organism->inputsUsed;
    // std::cout << "Bad Sensor: " << inputStartingNeuron << '\n';
}
AirSensor::AirSensor() {
    input = 1;
    output = 0;
}


std::unique_ptr<Module> AirSensor::Clone(Organism* newOrganism, NeuralNetwork* newNetwork) const {
    return std::make_unique<AirSensor>(newNetwork, newOrganism);
}
// Update method
void AirSensor::Update() {

    if (network == nullptr || organism == nullptr) {
        std::cerr << "HealthSensor not properly initialized!" << std::endl;
        return;
    }
    float usedAir = organism->air;
    if (organism->air <= 5000) {
        usedAir = 5000;
    }
    network->neurons[0 + inputStartingNeuron].activation = 5000 / usedAir;
    if (network->neurons[0 + inputStartingNeuron].activation > 1) {
        network->neurons[0 + inputStartingNeuron].activation = 1;
    }
    for (int i = 0; i < input; i++) {
        network->neurons[i + inputStartingNeuron].module = this;
    }
}
