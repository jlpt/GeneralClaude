
#include <iostream> // optional, for debugging output
#include "neural_network.h"
#include "health_sensor.h"
#include "organism.h"
// Constructor
HealthSensor::HealthSensor(NeuralNetwork* network, Organism* organism)
    : network{ network }, organism{ organism } {
    input = 1;
    output = 0;
    inputStartingNeuron = organism->inputsUsed;
   // std::cout << "Bad Sensor: " << inputStartingNeuron << '\n';
}
HealthSensor::HealthSensor() {
    input = 1;
    output = 0;
}


std::unique_ptr<Module> HealthSensor::Clone(Organism* newOrganism, NeuralNetwork* newNetwork) const {
    return std::make_unique<HealthSensor>(newNetwork, newOrganism);
}
// Update method
void HealthSensor::Update() {
    
    if (network == nullptr || organism == nullptr) {
        std::cerr << "HealthSensor not properly initialized!" << std::endl;
        return;
    }
    float usedHealth = organism->health;
    if (organism->health <= 1500) {
        usedHealth = 1500;
    }
    network->neurons[0 + inputStartingNeuron].activation = 1500 / usedHealth;
    if (network->neurons[0 + inputStartingNeuron].activation > 1) {
        network->neurons[0 + inputStartingNeuron].activation = 1;
    }
    for (int i = 0; i < input; i++) {
        network->neurons[i + inputStartingNeuron].module = this;
    }
}
