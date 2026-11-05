#include "neural_network.h"

#include <algorithm>
#include "config.h"
#include "random.h"

namespace {

// Diagram layout (0..1): inputs on the left, outputs on the right, hidden
// neurons scattered in between.
constexpr float kInputColumn = 0.05f;
constexpr float kOutputColumn = 0.95f;

float EvenlySpaced(int i, int count) { return count > 1 ? float(i) / float(count - 1) : 0.5f; }

Neuron MakeHidden(const Config& cfg) {
    Neuron n;
    n.type = NeuronType::Hidden;
    n.threshold = rng::Float(cfg.minThreshold, cfg.maxThreshold);
    n.refractoryPeriod = uint8_t(rng::Int(cfg.minRefractory, cfg.maxRefractory));
    n.layoutX = rng::Float(0.25f, 0.75f);
    n.layoutY = rng::Float(0.0f, 1.0f);
    return n;
}

}  // namespace

NeuralNetwork::NeuralNetwork(int numInputs, int numOutputs, int numHidden, int numConnections,
                             const Config& cfg)
    : numInputs_(numInputs), numOutputs_(numOutputs) {
    neurons_.reserve(size_t(numInputs + numOutputs + numHidden));
    for (int i = 0; i < numInputs; ++i) {
        Neuron n;
        n.type = NeuronType::Input;
        n.layoutX = kInputColumn;
        n.layoutY = EvenlySpaced(i, numInputs);
        neurons_.push_back(n);
    }
    for (int i = 0; i < numOutputs; ++i) {
        Neuron n;
        n.type = NeuronType::Output;
        n.layoutX = kOutputColumn;
        n.layoutY = EvenlySpaced(i, numOutputs);
        neurons_.push_back(n);
    }
    for (int i = 0; i < numHidden; ++i) neurons_.push_back(MakeHidden(cfg));

    inputDrive_.assign(size_t(numInputs), 0.0f);
    incoming_.assign(neurons_.size(), 0.0f);

    // AddConnection skips self-loops, so allow a few retries.
    for (int added = 0, attempts = 0; added < numConnections && attempts < numConnections * 4; ++attempts) {
        if (AddConnection()) ++added;
    }
}

void NeuralNetwork::SetInput(int index, float drive) {
    inputDrive_[size_t(index)] = std::clamp(drive, 0.0f, 1.0f);
}

float NeuralNetwork::Input(int index) const { return inputDrive_[size_t(index)]; }

float NeuralNetwork::Output(int index) const { return neurons_[size_t(numInputs_ + index)].potential; }

void NeuralNetwork::Step(const Config& cfg) {
    // 1. Deliver the spikes fired on the previous step. Doing this as a
    //    separate pass means update order doesn't matter.
    std::fill(incoming_.begin(), incoming_.end(), 0.0f);
    for (const Neuron& n : neurons_) {
        if (!n.spiked) continue;
        for (const Connection& c : n.connections) incoming_[c.Target()] += c.Weight();
    }

    // 2. Integrate and fire.
    for (size_t i = 0; i < neurons_.size(); ++i) {
        Neuron& n = neurons_[i];
        n.spiked = false;
        switch (n.type) {
        case NeuronType::Input:
            // Rate coding: a drive of 1 fires every step, 0.25 every 4th step.
            n.potential += inputDrive_[i];
            if (n.potential >= 1.0f) {
                n.potential -= 1.0f;
                n.spiked = true;
            }
            break;
        case NeuronType::Output:
            n.potential = std::clamp(n.potential * cfg.outputDecay + incoming_[i], 0.0f, 1.0f);
            break;
        case NeuronType::Hidden:
            if (n.refractory > 0) {
                --n.refractory;
                n.potential = 0.0f;
                break;
            }
            n.potential = std::max(-1.0f, n.potential * cfg.membraneLeak + incoming_[i]);
            if (n.potential >= n.threshold) {
                n.spiked = true;
                n.potential = 0.0f;
                n.refractory = n.refractoryPeriod;
            }
            break;
        }
    }
}

void NeuralNetwork::ResetState() {
    for (Neuron& n : neurons_) {
        n.potential = 0.0f;
        n.spiked = false;
        n.refractory = 0;
    }
    std::fill(inputDrive_.begin(), inputDrive_.end(), 0.0f);
}

size_t NeuralNetwork::MemoryBytes() const {
    size_t bytes = sizeof(*this) + neurons_.capacity() * sizeof(Neuron) +
                   (inputDrive_.capacity() + incoming_.capacity()) * sizeof(float);
    for (const Neuron& n : neurons_) bytes += n.connections.capacity() * sizeof(Connection);
    return bytes;
}

// --- Mutation ---------------------------------------------------------------

void NeuralNetwork::Mutate(int mutations, const Config& cfg) {
    const float addNeuron = cfg.addNeuronChance;
    const float removeNeuron = addNeuron + cfg.removeNeuronChance;
    const float addConnection = removeNeuron + cfg.addConnectionChance;
    const float removeConnection = addConnection + cfg.removeConnectionChance;
    const float threshold = removeConnection + cfg.thresholdChance;

    for (int i = 0; i < mutations; ++i) {
        const float roll = rng::Float(0.0f, 1.0f);
        if (roll < addNeuron) AddNeuron(cfg);
        else if (roll < removeNeuron) RemoveNeuron();
        else if (roll < addConnection) AddConnection();
        else if (roll < removeConnection) RemoveConnection();
        else if (roll < threshold) ModifyThreshold(cfg);
        else ModifyWeight(cfg);
    }
}

int NeuralNetwork::RandomSource() const {
    const int hidden = int(neurons_.size() - FirstHidden());
    const int choice = rng::Int(0, numInputs_ + hidden - 1);
    return choice < numInputs_ ? choice : choice + numOutputs_;  // skip over the outputs
}

int NeuralNetwork::RandomTarget() const { return rng::Int(numInputs_, int(neurons_.size()) - 1); }

bool NeuralNetwork::PickConnection(size_t& neuron, size_t& index) const {
    if (synapseCount_ == 0) return false;
    size_t remaining = size_t(rng::Int(0, int(synapseCount_) - 1));
    for (size_t i = 0; i < neurons_.size(); ++i) {
        const size_t count = neurons_[i].connections.size();
        if (remaining < count) {
            neuron = i;
            index = remaining;
            return true;
        }
        remaining -= count;
    }
    return false;
}

bool NeuralNetwork::AddConnection() {
    const bool hasSource = numInputs_ > 0 || neurons_.size() > FirstHidden();
    const bool hasTarget = neurons_.size() > size_t(numInputs_);
    if (!hasSource || !hasTarget) return false;

    const int source = RandomSource();
    const int target = RandomTarget();
    if (source == target) return false;  // no self-loops
    neurons_[size_t(source)].connections.emplace_back(uint32_t(target), rng::Float(-1.0f, 1.0f));
    ++synapseCount_;
    return true;
}

void NeuralNetwork::RemoveConnection() {
    size_t neuron, index;
    if (!PickConnection(neuron, index)) return;
    auto& connections = neurons_[neuron].connections;
    connections.erase(connections.begin() + std::ptrdiff_t(index));
    --synapseCount_;
}

void NeuralNetwork::ModifyWeight(const Config& cfg) {
    size_t neuron, index;
    if (!PickConnection(neuron, index)) return;
    Connection& c = neurons_[neuron].connections[index];
    c.SetWeight(c.Weight() + rng::Float(-cfg.weightStep, cfg.weightStep));
}

void NeuralNetwork::AddNeuron(const Config& cfg) {
    if (neurons_.size() + 1 >= Connection::kMaxNeurons) return;
    const uint32_t newIndex = uint32_t(neurons_.size());
    Neuron added = MakeHidden(cfg);

    size_t from, index;
    if (PickConnection(from, index)) {
        // Split an existing synapse A->B into A->new->B. The new neuron
        // relays A's spikes, so the old pathway keeps working.
        Connection& old = neurons_[from].connections[index];
        added.connections.emplace_back(old.Target(), old.Weight());
        old.SetTarget(newIndex);
        old.SetWeight(1.0f);
        ++synapseCount_;
    }
    neurons_.push_back(std::move(added));
    incoming_.push_back(0.0f);
}

void NeuralNetwork::RemoveNeuron() {
    if (neurons_.size() <= FirstHidden()) return;  // only hidden neurons can be removed
    const uint32_t victim = uint32_t(rng::Int(int(FirstHidden()), int(neurons_.size()) - 1));
    const uint32_t last = uint32_t(neurons_.size() - 1);

    // Move the last neuron into the victim's slot, then fix up the synapses
    // that pointed at either of them.
    synapseCount_ -= neurons_[victim].connections.size();
    if (victim != last) neurons_[victim] = std::move(neurons_[last]);
    neurons_.pop_back();
    incoming_.pop_back();

    for (Neuron& n : neurons_) {
        auto& connections = n.connections;
        const size_t before = connections.size();
        connections.erase(std::remove_if(connections.begin(), connections.end(),
                                         [victim](const Connection& c) { return c.Target() == victim; }),
                          connections.end());
        synapseCount_ -= before - connections.size();
        if (victim != last) {
            for (Connection& c : connections) {
                if (c.Target() == last) c.SetTarget(victim);
            }
        }
    }
}

void NeuralNetwork::ModifyThreshold(const Config& cfg) {
    if (neurons_.size() <= FirstHidden()) return;
    Neuron& n = neurons_[size_t(rng::Int(int(FirstHidden()), int(neurons_.size()) - 1))];
    n.threshold = std::clamp(n.threshold + rng::Float(-cfg.thresholdStep, cfg.thresholdStep),
                             cfg.minThreshold, cfg.maxThreshold);
}
