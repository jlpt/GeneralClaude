#include "headless.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include "random.h"
#include "world.h"

bool HasFlag(int argc, char** argv, const char* flag) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], flag) == 0) return true;
    }
    return false;
}

namespace {

// Reads "--name N". Returns false (and prints why) on a bad number.
bool ReadNumber(int argc, char** argv, const char* flag, unsigned long long& value) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], flag) != 0) continue;
        if (i + 1 >= argc) {
            std::fprintf(stderr, "%s needs a number\n", flag);
            return false;
        }
        try {
            value = std::stoull(argv[i + 1]);
        } catch (const std::exception&) {
            std::fprintf(stderr, "%s needs a number, got '%s'\n", flag, argv[i + 1]);
            return false;
        }
    }
    return true;
}

void PrintHeader() {
    std::printf("%11s %5s %7s %5s | %7s %7s %8s %8s | %8s %8s | %7s %6s %7s %6s\n", "step", "alive", "max gen",
                "diff", "neurons", "biggest", "synapses", "biggest", "births", "spawned", "poison%", "learn",
                "shared", "eaten");
}

// Poison bites as a share of all bites since the previous row.
struct BiteCounter {
    uint64_t poison = 0, good = 0;
    double ShareSince(const WorldStats& s) {
        const uint64_t p = s.poisonBites - poison, g = s.goodBites - good;
        poison = s.poisonBites;
        good = s.goodBites;
        return p + g ? 100.0 * double(p) / double(p + g) : 0.0;
    }
};

void PrintRow(const World& world, BiteCounter& bites) {
    const size_t population = world.organisms.size();
    const double neurons = population ? double(world.TotalNeurons()) / double(population) : 0.0;
    const double synapses = population ? double(world.TotalSynapses()) / double(population) : 0.0;
    size_t biggestNeurons = 0, biggestSynapses = 0;
    double learningRate = 0.0;
    for (const Organism& o : world.organisms) {
        if (o.brain.NeuronCount() > biggestNeurons) biggestNeurons = o.brain.NeuronCount();
        if (o.brain.SynapseCount() > biggestSynapses) biggestSynapses = o.brain.SynapseCount();
        learningRate += o.brain.LearningRate();
    }
    if (population) learningRate /= double(population);
    std::printf("%11llu %5zu %7d %4.0f%% | %7.1f %7zu %8.1f %8zu | %8llu %8llu | %6.1f%% %6.3f %7llu %6llu\n",
                (unsigned long long)world.stats.steps, population, world.MaxGeneration(), world.difficulty * 100.0f,
                neurons, biggestNeurons, synapses, biggestSynapses, (unsigned long long)world.stats.births,
                (unsigned long long)world.stats.spawned, bites.ShareSince(world.stats), learningRate,
                (unsigned long long)world.stats.bigFoodShared, (unsigned long long)world.stats.eatenByPredators);
    std::fflush(stdout);
}

}  // namespace

int RunHeadless(int argc, char** argv) {
    unsigned long long steps = 1000000, report = 50000, seed = 0;
    if (!ReadNumber(argc, argv, "--steps", steps) || !ReadNumber(argc, argv, "--report", report) ||
        !ReadNumber(argc, argv, "--seed", seed)) {
        return 1;
    }
    if (report == 0) report = steps;

    Config cfg;
    cfg.seed = uint32_t(seed);
    if (HasFlag(argc, argv, "--free-brains")) cfg.brainCostPerNeuron = cfg.brainCostPerSynapse = 0.0f;
    if (HasFlag(argc, argv, "--no-vision")) cfg.vision = false;
    if (HasFlag(argc, argv, "--no-springs")) cfg.springs = false;
    if (HasFlag(argc, argv, "--no-seasons")) cfg.seasons = false;
    const bool stage3 = HasFlag(argc, argv, "--stage3");  // everything from stage 4 off
    if (stage3 || HasFlag(argc, argv, "--no-signals")) cfg.signals = false;
    if (stage3 || HasFlag(argc, argv, "--no-coop-food")) cfg.cooperativeFood = false;
    if (stage3 || HasFlag(argc, argv, "--no-predators")) cfg.predators = false;
    if (stage3 || HasFlag(argc, argv, "--no-poison")) cfg.poison = false;
    if (stage3 || HasFlag(argc, argv, "--no-plasticity")) cfg.plasticity = false;
    cfg.organismSensor = HasFlag(argc, argv, "--organism-sensor");
    cfg.positionSensor = HasFlag(argc, argv, "--position-sensor");
    cfg.slices = HasFlag(argc, argv, "--slices");

    World world(cfg);
    std::printf("Nworks headless run: %llu steps, seed %u\n", steps, rng::CurrentSeed());
    auto onOff = [](bool on) { return on ? "on" : "off"; };
    std::printf("brain cost %s, vision %s, springs %s, seasons %s, signals %s, big food %s, predators %s, "
                "poison %s, learning %s\n",
                onOff(cfg.brainCostPerNeuron > 0.0f || cfg.brainCostPerSynapse > 0.0f), onOff(cfg.vision),
                onOff(cfg.springs), onOff(cfg.seasons), onOff(cfg.signals), onOff(cfg.cooperativeFood),
                onOff(cfg.predators), onOff(cfg.poison), onOff(cfg.plasticity));
    PrintHeader();

    BiteCounter bites;
    const auto start = std::chrono::steady_clock::now();
    for (unsigned long long i = 1; i <= steps; ++i) {
        world.Step();
        if (i % report == 0 || i == steps) PrintRow(world, bites);
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::printf("Done in %.1f s (%.0f steps/s). Largest brain ever: %zu neurons.\n", seconds,
                seconds > 0.0 ? double(steps) / seconds : 0.0, world.stats.largestBrainEver);
    std::printf("diff = curriculum difficulty; neurons/synapses = average per organism; biggest = largest brain "
                "alive;\npoison%% = poisonous share of bites since the last row (50%% = no better than chance); "
                "learn = average learning rate;\nshared = big food opened together; eaten = killed by predators.\n");
    return 0;
}
