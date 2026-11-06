#pragma once
#include <cstdint>
#include <memory>
#include <vector>
#include "module.h"
#include "neural_network.h"

struct Config;
class World;

struct Rgb {
    uint8_t r = 255, g = 255, b = 255;
};

class Organism {
public:
    Organism(uint64_t id, float x, float y, const Config& cfg);
    Organism(Organism&&) noexcept = default;
    Organism& operator=(Organism&&) noexcept = default;
    Organism(const Organism&) = delete;
    Organism& operator=(const Organism&) = delete;

    // Adds a module and gives it the next free block of inputs/outputs.
    // Add all modules before creating the brain.
    void AddModule(std::unique_ptr<Module> module);
    int InputsUsed() const { return inputsUsed_; }
    int OutputsUsed() const { return outputsUsed_; }

    // One step of life: sense -> think -> act -> burn health and air.
    void Update(World& world);

    // A mutated copy of this organism.
    Organism MakeOffspring(uint64_t childId, const Config& cfg) const;

    uint64_t id;
    float x, y;
    float radius;
    float health;
    float air;
    int foodEaten = 0;     // since the last reproduction
    int generation = 0;
    float mutationRate = 1.0f;
    float signal = 0.0f;   // 0..1, what the SignalModule is broadcasting
    float eatSignal = 1.0f;  // 0..1, the EatModule output (eatAction: eats only at 0.5 or more)
    float reward = 0.0f;   // + after good food, - after poison; fades every step
    int goodBitesInLife = 0, poisonBitesInLife = 0;  // for experiments
    int age = 0;           // steps lived
    bool alive = true;
    Rgb color;             // drifts slowly, so related organisms look alike
    NeuralNetwork brain;
    std::vector<std::unique_ptr<Module>> modules;

private:
    int inputsUsed_ = 0;
    int outputsUsed_ = 0;
};
