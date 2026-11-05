
#include <iostream> // optional, for debugging output
#include "neural_network.h"
#include "badfood_sensor.h"
#include "organism.h"
// Constructor
BadFoodSensor::BadFoodSensor(NeuralNetwork* network, Organism* organism, std::vector<BadFood>* foodList)
    : network{ network }, organism{ organism }, foodList{ foodList } {
    input = 5;
    output = 0;
    inputStartingNeuron = organism->inputsUsed;
   

   // std::cout << "Bad Sensor: " << inputStartingNeuron << '\n';

}
BadFoodSensor::BadFoodSensor() {
    input = 5;
    output = 0;
}


std::unique_ptr<Module> BadFoodSensor::Clone(Organism* newOrganism, NeuralNetwork* newNetwork) const {
    return std::make_unique<BadFoodSensor>(newNetwork, newOrganism, foodList);
}
// Update method
void BadFoodSensor::Update() {
    if (network == nullptr || organism == nullptr || foodList == nullptr) {
        std::cerr << "ModuleSensor not properly initialized!" << std::endl;
        return;
    }

    // --- Collision detection: remove any food the organism touches ---
    for (auto it = foodList->begin(); it != foodList->end(); ) {
        float dx = it->x - organism->x;
        float dy = it->y - organism->y;
        float distance = std::sqrt(dx * dx + dy * dy);
        if (distance < organism->size + it->size) {
            // Food is touched/eaten – remove it.
            organism->air = 10000;
          //  organism->foodEaten++;
            it = foodList->erase(it);
        }
        else {
            ++it;
        }
    }

    // --- Food sensing ---
    const float sensorRange = 500.0f;
    BadFood* closestFood = nullptr;
    float closestDist = sensorRange;

    for (BadFood& f : *foodList) {
        float dx = f.x - organism->x;
        float dy = f.y - organism->y;
        float distance = std::sqrt(dx * dx + dy * dy);
        if (distance < closestDist) {
            closestDist = distance;
            closestFood = &f;
        }
    }

    // Initialize baseline activations for all four directional neurons.
    // We assume the network has at least four neurons ordered as:
    // 0: left, 1: right, 2: up, 3: down.
    float activationLeft = 0.0f;
    float activationRight = 0.0f;
    float activationUp = 0.0f;
    float activationDown = 0.0f;

    if (closestFood != nullptr) {
        float deltaX = closestFood->x - organism->x;
        float deltaY = closestFood->y - organism->y;
        network->neurons[4 + inputStartingNeuron].activation = 0;
        // std::cout << organism->inputsUsed << '\n';
         // Calculate activations based on the distance along X.
         // If deltaX is negative, food is to the left; if positive, it's to the right.
        if (deltaX < 0) {
            activationLeft = (sensorRange - std::fabs(deltaX)) / sensorRange;
            if (activationLeft > 1.0f) activationLeft = 1.0f;
        }
        else {
            activationRight = (sensorRange - std::fabs(deltaX)) / sensorRange;
            if (activationRight > 1.0f) activationRight = 1.0f;
        }

        // Calculate activations based on the distance along Y.
        // If deltaY is negative, food is upward; if positive, it's downward.
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
   // std::cout << "food: " << inputStartingNeuron << '\n';
    // Update the network's neurons.

    network->neurons[0 + inputStartingNeuron].activation = activationLeft;
    network->neurons[1 + inputStartingNeuron].activation = activationRight;
    network->neurons[2 + inputStartingNeuron].activation = activationUp;
    network->neurons[3 + inputStartingNeuron].activation = activationDown;
   // std::cout << "Bad Sensor: " << inputStartingNeuron << '\n';
    for (int i = 0; i < input; i++) {
        network->neurons[i + inputStartingNeuron].module = this;
    }
  
}
