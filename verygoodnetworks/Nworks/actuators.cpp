#include "actuators.h"

#include "organism.h"
#include "world.h"

void MovementModule::Act(Organism& self, World& world) {
    const NeuralNetwork& brain = self.brain;
    const float speed = world.cfg.moveSpeed;
    self.x += (brain.Output(firstOutput + 0) - brain.Output(firstOutput + 1)) * speed;
    self.y += (brain.Output(firstOutput + 2) - brain.Output(firstOutput + 3)) * speed;
}

void SignalModule::Act(Organism& self, World& world) {
    self.signal = self.brain.Output(firstOutput);
    self.health -= world.cfg.signalCost * self.signal;
}

void EatModule::Act(Organism& self, World& /*world*/) { self.eatSignal = self.brain.Output(firstOutput); }

void SliceModule::Act(Organism& self, World& world) {
    constexpr float kReach = 600.0f;  // an output of 0 or 1 puts the endpoint 300 px away
    constexpr float kActive = 0.001f;

    if (cooldown_ > 0) {
        --cooldown_;
        return;
    }
    float out[4];
    for (int i = 0; i < 4; ++i) {
        out[i] = self.brain.Output(firstOutput + i);
        if (out[i] < kActive) return;
    }
    world.slices.push_back(Slice{self.x + (out[0] - 0.5f) * kReach, self.y + (out[1] - 0.5f) * kReach,
                                 self.x + (out[2] - 0.5f) * kReach, self.y + (out[3] - 0.5f) * kReach});
    cooldown_ = world.cfg.sliceCooldown;
}
