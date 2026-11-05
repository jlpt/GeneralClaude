#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H
#include "neuron.h"
#include <vector>

#include "food_sensor.h"
class NeuralNetwork {
public:
    int neuron_count;
    int synapse_count;
    std::vector<Neuron> neurons;
    std::vector<int> outputIndices;
    int nextNeuronId;
    NeuralNetwork();
    NeuralNetwork(int numNeurons, int numConnections, int numInput, int numOutput);
    NeuralNetwork(const NeuralNetwork& other);  // <-- Deep copy constructor
    void Draw();
    void GetSize();
    void Update();
    void Mutate(int power);
    void ModifyWeight();
    void AddWeight();
    void AddNeuron();
    void RemoveWeight();
    void RemoveNeuron();
 
};

#endif // NEURAL_NETWORK_H
