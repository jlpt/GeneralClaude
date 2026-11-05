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
    if (cfg.plasticity) learningRate_ = rng::Float(0.0f, cfg.maxInitialLearningRate);

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

float NeuralNetwork::LearnedChange(size_t neuron, size_t connection) const {
    if (neuron >= plastic_.size() || connection >= plastic_[neuron].size()) return 0.0f;
    return plastic_[neuron][connection].learned;
}

void NeuralNetwork::PrepareScratch(bool plastic) {
    if (incoming_.size() != neurons_.size()) incoming_.resize(neurons_.size());
    std::fill(incoming_.begin(), incoming_.end(), 0.0f);
    fired_.resize(neurons_.size());
    if (!plastic) return;
    plastic_.resize(neurons_.size());
    for (size_t i = 0; i < neurons_.size(); ++i) {
        if (plastic_[i].size() != neurons_[i].connections.size()) plastic_[i].assign(neurons_[i].connections.size(), {});
    }
}

void NeuralNetwork::Step(const Config& cfg) {
    const bool plastic = cfg.plasticity;
    PrepareScratch(plastic);
    ++clock_;

    // 1. Deliver the spikes fired on the previous step. Doing this as a
    //    separate pass means update order doesn't matter.
    for (size_t i = 0; i < neurons_.size(); ++i) {
        const Neuron& n = neurons_[i];
        fired_[i] = n.spiked ? 1 : 0;
        if (!n.spiked) continue;
        for (size_t k = 0; k < n.connections.size(); ++k) {
            const Connection& c = n.connections[k];
            float weight = c.Weight();
            if (plastic) weight = std::clamp(weight + plastic_[i][k].learned, -1.0f, 1.0f);
            incoming_[c.Target()] += weight;
        }
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

    // 3. Learn.
    if (plastic && learningRate_ > 0.0f) Learn(cfg);
}

namespace {

// Eligibility older than this many steps counts as zero.
constexpr uint32_t kTraceWindow = 64;
// Forgetting is applied in batches every this many steps.
constexpr uint32_t kForgetEvery = 64;

// decay^age for age < kTraceWindow, cached for the current decay value.
const float* DecayTable(float decay) {
    static float cachedDecay = -1.0f;
    static float table[kTraceWindow];
    if (decay != cachedDecay) {
        cachedDecay = decay;
        float value = 1.0f;
        for (uint32_t age = 0; age < kTraceWindow; ++age, value *= decay) table[age] = value;
    }
    return table;
}

}  // namespace

void NeuralNetwork::Learn(const Config& cfg) {
    // Three-factor rule: a synapse becomes eligible when its source spiked
    // just before its target responded; the reward signal then turns
    // eligibility into a lasting (but slowly fading) weight change.
    // Work is only done for spikes, during rewards, and every kForgetEvery steps.
    const float* decay = DecayTable(cfg.traceDecay);
    auto eligibility = [&](const Plastic& p) {
        const uint32_t age = clock_ - p.traceStep;
        return age < kTraceWindow ? p.trace * decay[age] : 0.0f;
    };

    for (size_t i = 0; i < neurons_.size(); ++i) {
        if (!fired_[i]) continue;
        const auto& connections = neurons_[i].connections;
        for (size_t k = 0; k < connections.size(); ++k) {
            const Neuron& target = neurons_[connections[k].Target()];
            const float response = target.type == NeuronType::Output ? target.potential : (target.spiked ? 1.0f : 0.0f);
            if (response <= 0.0f) continue;
            Plastic& p = plastic_[i][k];
            p.trace = eligibility(p) + response;
            p.traceStep = clock_;
        }
    }

    if (std::fabs(reward_) > 1e-3f) {
        const float scale = learningRate_ * reward_;
        for (auto& synapses : plastic_) {
            for (Plastic& p : synapses) {
                const float e = eligibility(p);
                if (e > 0.0f) p.learned = std::clamp(p.learned + scale * e, -1.0f, 1.0f);
            }
        }
    }

    if (clock_ % kForgetEvery == 0) {
        const float keep = std::pow(1.0f - cfg.learnedForgetting, float(kForgetEvery));
        for (auto& synapses : plastic_) {
            for (Plastic& p : synapses) p.learned *= keep;
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
    plastic_.clear();  // learning is not inherited
    reward_ = 0.0f;
    clock_ = 0;
}

size_t NeuralNetwork::MemoryBytes() const {
    size_t bytes = sizeof(*this) + neurons_.capacity() * sizeof(Neuron) +
                   (inputDrive_.capacity() + incoming_.capacity()) * sizeof(float) + fired_.capacity();
    for (const Neuron& n : neurons_) bytes += n.connections.capacity() * sizeof(Connection);
    for (const auto& p : plastic_) bytes += p.capacity() * sizeof(Plastic);
    return bytes;
}

// --- Mutation ---------------------------------------------------------------

void NeuralNetwork::Mutate(int mutations, const Config& cfg) {
    const float addNeuron = cfg.addNeuronChance;
    const float removeNeuron = addNeuron + cfg.removeNeuronChance;
    const float addConnection = removeNeuron + cfg.addConnectionChance;
    const float removeConnection = addConnection + cfg.removeConnectionChance;
    const float threshold = removeConnection + cfg.thresholdChance;
    const float learning = threshold + (cfg.plasticity ? cfg.learningRateChance : 0.0f);

    for (int i = 0; i < mutations; ++i) {
        const float roll = rng::Float(0.0f, 1.0f);
        if (roll < addNeuron) AddNeuron(cfg);
        else if (roll < removeNeuron) RemoveNeuron();
        else if (roll < addConnection) AddConnection();
        else if (roll < removeConnection) RemoveConnection();
        else if (roll < threshold) ModifyThreshold(cfg);
        else if (roll < learning) ModifyLearningRate(cfg);
        else ModifyWeight(cfg);
    }
}

void NeuralNetwork::ModifyLearningRate(const Config& cfg) {
    learningRate_ = std::clamp(learningRate_ + rng::Float(-cfg.learningRateStep, cfg.learningRateStep), 0.0f, 0.5f);
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
