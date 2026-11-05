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
    void ResetState();                      // clear potentials, spikes and anything learned
    void Mutate(int mutations, const Config& cfg);

    // Lifetime learning (Config::plasticity): positive reward strengthens
    // recently eligible synapses, negative reward weakens them.
    void SetReward(float reward) { reward_ = reward; }
    float LearningRate() const { return learningRate_; }
    float LearnedChange(size_t neuron, size_t connection) const;  // 0 without plasticity

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
    void ModifyLearningRate(const Config& cfg);
    void PrepareScratch(bool plastic);
    void Learn(const Config& cfg);

    // Per-synapse learning state, parallel to each neuron's connections.
    // Traces decay lazily: `trace` is the value at step `traceStep`.
    struct Plastic {
        float trace = 0.0f;      // eligibility: recent pre-before-post pairings
        float learned = 0.0f;    // change added to the inherited weight
        uint32_t traceStep = 0;
    };

    std::vector<Neuron> neurons_;
    std::vector<float> inputDrive_;  // sensor value for each input neuron
    std::vector<float> incoming_;    // scratch: spike input arriving this step
    std::vector<uint8_t> fired_;     // scratch: which neurons spiked on the previous step
    std::vector<std::vector<Plastic>> plastic_;  // only used with plasticity
    int numInputs_ = 0;
    int numOutputs_ = 0;
    size_t synapseCount_ = 0;
    float learningRate_ = 0.0f;      // a gene
    float reward_ = 0.0f;
    uint32_t clock_ = 0;             // steps since the last ResetState
};
