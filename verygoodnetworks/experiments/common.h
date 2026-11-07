// Helpers shared by the experiment tools (nworks_exp, nworks_iq, the Python bridge).
#pragma once
#include <map>
#include <memory>
#include <string>
#include "organism.h"
#include "world.h"

// Sets one Config field from "key=value" text. Returns false for an unknown key.
inline bool ApplyOverride(Config& c, const std::string& key, const std::string& value) {
    std::map<std::string, float*> floats = {
        {"maxHealth", &c.maxHealth}, {"maxAir", &c.maxAir}, {"moveSpeed", &c.moveSpeed},
        {"brainCostPerNeuron", &c.brainCostPerNeuron}, {"brainCostPerSynapse", &c.brainCostPerSynapse},
        {"foodSpawnChance", &c.foodSpawnChance}, {"poisonDamage", &c.poisonDamage},
        {"traceDecay", &c.traceDecay}, {"learnedForgetting", &c.learnedForgetting}, {"rewardDecay", &c.rewardDecay},
        {"maxInitialLearningRate", &c.maxInitialLearningRate}, {"learningRateStep", &c.learningRateStep},
        {"learningRateChance", &c.learningRateChance}, {"mutationFraction", &c.mutationFraction},
        {"worldWidth", &c.worldWidth}, {"worldHeight", &c.worldHeight}, {"visionRange", &c.visionRange},
        {"predatorBaseSpeed", &c.predatorBaseSpeed}, {"predatorExtraSpeed", &c.predatorExtraSpeed},
        {"signalCost", &c.signalCost}, {"foodNearSight", &c.foodNearSight}, {"randomRefillShare", &c.randomRefillShare}, {"poisonShare", &c.poisonShare}, {"springRecharge", &c.springRecharge}, {"curriculumStep", &c.curriculumStep},
    };
    std::map<std::string, int*> ints = {
        {"minPopulation", &c.minPopulation}, {"maxAge", &c.maxAge}, {"minFood", &c.minFood}, {"foodToReproduce", &c.foodToReproduce}, {"maxFood", &c.maxFood},
        {"initialFood", &c.initialFood}, {"foodVisibleSteps", &c.foodVisibleSteps}, {"poisonSwapInterval", &c.poisonSwapInterval},
        {"seasonLength", &c.seasonLength}, {"foodLifetime", &c.foodLifetime}, {"curriculumWindow", &c.curriculumWindow},
        {"initialHiddenNeurons", &c.initialHiddenNeurons}, {"initialConnections", &c.initialConnections},
        {"predatorCount", &c.predatorCount}, {"survivorArchiveSize", &c.survivorArchiveSize},
    };
    std::map<std::string, bool*> bools = {
        {"vision", &c.vision}, {"springs", &c.springs}, {"seasons", &c.seasons}, {"signals", &c.signals},
        {"cooperativeFood", &c.cooperativeFood}, {"predators", &c.predators}, {"poison", &c.poison},
        {"plasticity", &c.plasticity}, {"poisonScalesWithDifficulty", &c.poisonScalesWithDifficulty},
        {"colourSmell", &c.colourSmell}, {"learningHasEffect", &c.learningHasEffect},
        {"learnFromReward", &c.learnFromReward}, {"learnFromPunishment", &c.learnFromPunishment},
        {"plasticInputsOnly", &c.plasticInputsOnly}, {"foodCurriculum", &c.foodCurriculum},
        {"eatAction", &c.eatAction}, {"separateFoodCaps", &c.separateFoodCaps}, {"refillFromSurvivors", &c.refillFromSurvivors},
    };
    if (auto it = floats.find(key); it != floats.end()) { *it->second = std::stof(value); return true; }
    if (auto it = ints.find(key); it != ints.end()) { *it->second = std::stoi(value); return true; }
    if (auto it = bools.find(key); it != bools.end()) { *it->second = value == "1" || value == "true" || value == "on"; return true; }
    return false;
}

// An exact copy of a creature: MakeOffspring with mutation strength 0. The
// copy's mutation rate is 0 as well; reset it if the copy should evolve.
inline Organism CloneOf(Organism& src, uint64_t id, const Config& cfg) {
    const float saved = src.mutationRate;
    src.mutationRate = 0.0f;
    Organism c = src.MakeOffspring(id, cfg);
    src.mutationRate = saved;
    c.generation = src.generation;
    return c;
}

// Applies every "key=value" in a space-separated string. Returns false on an unknown key.
inline bool ApplyOverrides(Config& c, const std::string& text) {
    size_t pos = 0;
    while (pos < text.size()) {
        while (pos < text.size() && text[pos] == ' ') ++pos;
        size_t end = text.find(' ', pos);
        if (end == std::string::npos) end = text.size();
        const std::string item = text.substr(pos, end - pos);
        pos = end;
        if (item.empty()) continue;
        const size_t eq = item.find('=');
        if (eq == std::string::npos || !ApplyOverride(c, item.substr(0, eq), item.substr(eq + 1))) return false;
    }
    return true;
}

// A creature body (all the modules cfg asks for) whose outputs are set from
// outside instead of by a spiking brain: brain has inputs and outputs only.
inline Organism MakeExternalBody(const Config& cfg) {
    Config scratchCfg = cfg;
    scratchCfg.minPopulation = 1;
    scratchCfg.refillFromSurvivors = false;
    World scratch(scratchCfg);
    scratch.Step();
    Organism body = CloneOf(scratch.organisms[0], 1, cfg);
    body.brain = NeuralNetwork(body.brain.NumInputs(), body.brain.NumOutputs(), 0, 0, cfg);
    body.external = true;
    return body;
}
