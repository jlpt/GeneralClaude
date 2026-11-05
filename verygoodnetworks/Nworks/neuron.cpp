#include "neuron.h"
#include <random>
#include "raylib.h"
#include <iostream>
#include <string>
void Connection::setWeight(float f) {
    // Map f in [-1, 1] to an int8_t in [-127, 127]
    weight = static_cast<int8_t>(f * 127.0f);
}

float Connection::getWeight() const {
    return weight / 127.0f;
}

void Connection::setDestNeuron(uint32_t value) {
    destNeuron[0] = value & 0xFF;
    destNeuron[1] = (value >> 8) & 0xFF;
    destNeuron[2] = (value >> 16) & 0xFF;
}

uint32_t Connection::getDestNeuron() const {
    return destNeuron[0] | (destNeuron[1] << 8) | (destNeuron[2] << 16);
}

Connection::Connection(float weightVal, uint32_t destNeuronValue)
{
    // Set the weight using the provided value
    setWeight(weightVal);
    
    // Set the destination neuron using the provided 24-bit index
    setDestNeuron(destNeuronValue);
}


Neuron::Neuron() : x{ 0 }, y{ 0 } {}

Neuron::Neuron(float param_x, float param_y, uint8_t param_refactoryPeriod, NeuronType type, float thresholdMin, float thresholdMax)
    : x{ param_x }, y{ param_y }, refactoryPeriod{ param_refactoryPeriod }, type{ type }, thresholdMin{ thresholdMin }, thresholdMax{ thresholdMax } {
    module = nullptr;
}


void Neuron::update() {
    /*activation = activation * .99f;
    if (type != NeuronType::Output) {
        if (refactory > 0) {
            refactory--;
        }
        else {

            Propagate();
            refactory = refactoryPeriod;
        }
        
    }*/
}
void Neuron::Propagate(std::vector<Neuron>& neurons) {
    if (refactory <= 0 && activation >= thresholdMin && activation <= thresholdMax) {
        // Apply leaky ReLU before propagation
        float act = activation;
        if (act < 0.0f) act *= 0.01f;  // small negative slope
        if (act > 1.0f) act = 1.0f;

        for (Connection& c : connections) {
            uint32_t destNeuronIndex = c.getDestNeuron();
            if (destNeuronIndex >= neurons.size()) continue;

            Neuron& destNeuron = neurons[destNeuronIndex];
            destNeuron.activation += act * c.getWeight();
            // Clamp destination activation to allow slight inhibition
            destNeuron.activation = std::clamp(destNeuron.activation, -0.1f, 1.0f);
        }

        // Reset refactory, but keep activation to decay naturally
        refactory = refactoryPeriod;
    }
}

