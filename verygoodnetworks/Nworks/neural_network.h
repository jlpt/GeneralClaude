#pragma once
#include <cstddef>
#include <vector>
#include "neuron.h"

struct Config;

// A recurrent spiking network. Neurons are stored as
//   [inputs][outputs][hidden...]
// so input i is neuron i, output j is neuron NumInputs() + j, and hidden
// neurons can be added or removed at the end without moving the others.
class NeuralNetwork {
public:
    NeuralNetwork() = default;
    NeuralNetwork(int numInputs, int numOutputs, int numHidden, int numConnections, const Config& cfg);

    void SetInput(int index, float drive);  // 0..1
    float Input(int index) const;
    float Output(int index) const;          // 0..1

    void Step(const Config& cfg);           // advance one simulation step
    void ResetState();                      // clear potentials and spikes
    void Mutate(int mutations, const Config& cfg);

    int NumInputs() const { return numInputs_; }
    int NumOutputs() const { return numOutputs_; }
    size_t NeuronCount() const { return neurons_.size(); }
    size_t SynapseCount() const { return synapseCount_; }
    size_t MemoryBytes() const;
    const std::vector<Neuron>& Neurons() const { return neurons_; }

private:
    size_t FirstHidden() const { return size_t(numInputs_ + numOutputs_); }
    int RandomSource() const;  // an input or hidden neuron
    int RandomTarget() const;  // an output or hidden neuron
    bool PickConnection(size_t& neuron, size_t& index) const;  // uniform over all synapses

    bool AddConnection();
    void RemoveConnection();
    void ModifyWeight(const Config& cfg);
    void AddNeuron(const Config& cfg);
    void RemoveNeuron();
    void ModifyThreshold(const Config& cfg);

    std::vector<Neuron> neurons_;
    std::vector<float> inputDrive_;  // sensor value for each input neuron
    std::vector<float> incoming_;    // scratch: spike input arriving this step
    int numInputs_ = 0;
    int numOutputs_ = 0;
    size_t synapseCount_ = 0;
};
