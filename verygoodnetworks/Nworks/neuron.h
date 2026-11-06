#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

// A synapse packed into 4 bytes: a 24-bit target neuron index and an 8-bit
// weight in [-1, 1].
class Connection {
public:
    static constexpr uint32_t kMaxNeurons = 1u << 24;

    Connection(uint32_t target, float weight) {
        SetTarget(target);
        SetWeight(weight);
    }

    uint32_t Target() const {
        return uint32_t(target_[0]) | (uint32_t(target_[1]) << 8) | (uint32_t(target_[2]) << 16);
    }
    void SetTarget(uint32_t target) {
        target_[0] = uint8_t(target & 0xFF);
        target_[1] = uint8_t((target >> 8) & 0xFF);
        target_[2] = uint8_t((target >> 16) & 0xFF);
    }

    float Weight() const { return weight_ / 127.0f; }
    void SetWeight(float weight) {
        weight_ = int8_t(std::lround(std::clamp(weight, -1.0f, 1.0f) * 127.0f));
    }

private:
    uint8_t target_[3];
    int8_t weight_;
};
static_assert(sizeof(Connection) == 4, "Connection should pack into 4 bytes");

enum class NeuronType : uint8_t { Input, Output, Hidden };

// A leaky integrate-and-fire neuron.
//  - Input neurons charge up from their sensor value and spike at a rate
//    proportional to it.
//  - Hidden neurons leak, sum the weights of incoming spikes, spike when the
//    potential reaches the threshold, then reset and stay silent for
//    `refractoryPeriod` steps.
//  - Output neurons never spike; their potential is a smoothed 0..1 level
//    that the body reads.
struct Neuron {
    NeuronType type = NeuronType::Hidden;
    uint8_t refractoryPeriod = 0;
    uint8_t refractory = 0;   // silent steps remaining
    bool spiked = false;      // fired on the most recent step
    float potential = 0.0f;
    float threshold = 1.0f;
    float layoutX = 0.0f;     // position in the brain diagram, 0..1
    float layoutY = 0.0f;
    std::vector<Connection> connections;  // outgoing synapses
};
