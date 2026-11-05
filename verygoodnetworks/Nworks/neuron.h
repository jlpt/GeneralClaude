#ifndef NEURON_H
#define NEURON_H
#include <cstdint>
#include <vector>
#include "module.h"
#include <cmath>
struct Neuron; // Forward declaration for Connection
// Declaration in the header
class Connection {
public:
    uint8_t destNeuron[3];  // 3 bytes for a 24-bit index
    int8_t weight;          // Stored as an 8-bit quantized value
    float lastWeightDelta = 0.0f; // Added to track recent weight change

    Connection(float weightVal, uint32_t destNeuronValue);
    void setWeight(float f);
    float getWeight() const;
    void setDestNeuron(uint32_t value);
    uint32_t getDestNeuron() const;
};
inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::expf(-x));
}

struct Neuron {

    enum class NeuronType {
        Input,
        Output,
        Normal
    };

public:
    bool thresholdReached = false;
    int delay = 0;
    int delayTime = 1;
    NeuronType type;
    float thresholdMax;
    float thresholdMin;
    Module* module;
    float activation = 0.0f;
   
    float inputSum = 0.0f;  // collects weighted inputs before sigmoid
    float x, y;
    
uint8_t refactoryPeriod;
uint8_t refactory = 0;

std::vector<Connection> connections;

Neuron();
Neuron(float param_x, float param_y, uint8_t param_refactoryPeriod, NeuronType type, float thresholdMin, float thresholdMax);

void update();
void Propagate(std::vector<Neuron>& neurons);
};
#endif