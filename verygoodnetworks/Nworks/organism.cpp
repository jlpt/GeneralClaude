#include "organism.h"

#include <algorithm>
#include "config.h"
#include "random.h"
#include "world.h"

namespace {

uint8_t ShiftChannel(uint8_t value, int maxShift) {
    return uint8_t(std::clamp(int(value) + rng::Int(-maxShift, maxShift), 0, 255));
}

}  // namespace

Organism::Organism(uint64_t organismId, float startX, float startY, const Config& cfg)
    : id(organismId), x(startX), y(startY), radius(cfg.organismRadius), health(cfg.maxHealth), air(cfg.maxAir) {
    color = Rgb{uint8_t(rng::Int(0, 255)), uint8_t(rng::Int(0, 255)), uint8_t(rng::Int(0, 255))};
}

void Organism::AddModule(std::unique_ptr<Module> module) {
    module->firstInput = inputsUsed_;
    module->firstOutput = outputsUsed_;
    inputsUsed_ += module->InputCount();
    outputsUsed_ += module->OutputCount();
    modules.push_back(std::move(module));
}

void Organism::Update(World& world) {
    const Config& cfg = world.cfg;
    for (auto& module : modules) module->Sense(*this, world);
    brain.Step(cfg);
    for (auto& module : modules) module->Act(*this, world);

    x = std::clamp(x, 0.0f, cfg.worldWidth);
    y = std::clamp(y, 0.0f, cfg.worldHeight);
    health -= 1.0f + cfg.brainCostPerSynapse * float(brain.SynapseCount());
    air -= 1.0f;
}

Organism Organism::MakeOffspring(uint64_t childId, const Config& cfg) const {
    const float spread = cfg.offspringSpread;
    Organism child(childId, x + rng::Float(-spread, spread), y + rng::Float(-spread, spread), cfg);

    // A rare "macro mutation" changes the brain (and colour) much more.
    const bool macro = rng::Chance(cfg.macroMutationChance);
    const float strength = mutationRate * (macro ? cfg.macroMutationMultiplier : 1.0f);
    child.brain = brain;
    child.brain.Mutate(int(float(brain.SynapseCount()) * cfg.mutationFraction * strength), cfg);
    child.brain.ResetState();

    // The mutation rate itself is inherited and drifts, so it can evolve.
    child.mutationRate = mutationRate * (rng::Chance(0.5f) ? 1.0f + cfg.mutationRateDrift
                                                           : 1.0f - cfg.mutationRateDrift);
    child.generation = generation + 1;
    child.air = air;  // children inherit the parent's remaining air
    const int colorShift = macro ? 110 : 5;
    child.color = Rgb{ShiftChannel(color.r, colorShift), ShiftChannel(color.g, colorShift),
                      ShiftChannel(color.b, colorShift)};

    for (const auto& module : modules) child.AddModule(module->Clone());
    return child;
}
