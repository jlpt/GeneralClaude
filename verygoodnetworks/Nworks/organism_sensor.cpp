#include <iostream>
#include <cmath>
#include <memory>
#include <vector>
#include "organism_sensor.h"
#include "organism.h"

// Constructor: now organismList is a pointer.
OrganismSensor::OrganismSensor(NeuralNetwork* network, Organism* organism,
    std::vector<std::unique_ptr<Organism>>* organismList)
    : network{ network }, organism{ organism }, organismList{ organismList } {
    input = 5;
    output = 0;

    inputStartingNeuron = organism->inputsUsed;
   // std::cout << "Bad Sensor: " << inputStartingNeuron << '\n';
   
}

OrganismSensor::OrganismSensor()
    : network{ nullptr }, organism{ nullptr }, organismList{ nullptr } {
    input = 5;
    output = 0;
}

std::unique_ptr<Module> OrganismSensor::Clone(Organism* newOrganism, NeuralNetwork* newNetwork) const {
    // Pass the pointer to the same organism list.
    return std::make_unique<OrganismSensor>(newNetwork, newOrganism, organismList);
}

void OrganismSensor::Update() {
    if (network == nullptr || organism == nullptr || organismList == nullptr) {
        std::cerr << "OrganismSensor not properly initialized!" << std::endl;
        return;
    }
  
    // --- Organism (or food) sensing ---
    const float sensorRange = 600.0f;
    Organism* closestOrganism = nullptr;
    float closestDist = sensorRange;

    // Iterate over the vector of unique_ptrs
    for (auto& uptr : *organismList) {
        Organism* o = uptr.get();  // Get the raw pointer from the unique_ptr

        // Skip checking itself
        if (o == organism)
            continue;

        float dx = o->x - organism->x;
        float dy = o->y - organism->y;
        float distance = std::sqrt(dx * dx + dy * dy);
        if (distance < closestDist) {
            closestDist = distance;
            closestOrganism = o;
        }
    }

    // Use the sensed data to update the network's neurons.
    float activationLeft = 0.0f;
    float activationRight = 0.0f;
    float activationUp = 0.0f;
    float activationDown = 0.0f;

    if (closestOrganism != nullptr) {
        float deltaX = closestOrganism->x - organism->x;
        float deltaY = closestOrganism->y - organism->y;
        network->neurons[4 + inputStartingNeuron].activation = 0;
        if (deltaX < 0) {
            activationLeft = (sensorRange - std::fabs(deltaX)) / sensorRange;
            if (activationLeft > 1.0f) activationLeft = 1.0f;
        }
        else {
            activationRight = (sensorRange - std::fabs(deltaX)) / sensorRange;
            if (activationRight > 1.0f) activationRight = 1.0f;
        }
        if (deltaY < 0) {
            activationUp = (sensorRange - std::fabs(deltaY)) / sensorRange;
            if (activationUp > 1.0f) activationUp = 1.0f;
        }
        else {
            activationDown = (sensorRange - std::fabs(deltaY)) / sensorRange;
            if (activationDown > 1.0f) activationDown = 1.0f;
        }
    }
    else {
        network->neurons[4 + inputStartingNeuron].activation = 1;
    }
    //std::cout << "organism: " << inputStartingNeuron << '\n';
    network->neurons[0 + inputStartingNeuron].activation = activationLeft;
    network->neurons[1 + inputStartingNeuron].activation = activationRight;
    network->neurons[2 + inputStartingNeuron].activation = activationUp;
    network->neurons[3 + inputStartingNeuron].activation = activationDown;
    for (int i = 0; i < input; i++) {
        network->neurons[i + inputStartingNeuron].module = this;
    }
}
