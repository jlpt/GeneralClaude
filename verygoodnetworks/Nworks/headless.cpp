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

void PrintRow(const World& world) {
    const size_t population = world.organisms.size();
    const double neurons = population ? double(world.TotalNeurons()) / double(population) : 0.0;
    const double synapses = population ? double(world.TotalSynapses()) / double(population) : 0.0;
    std::printf("%11llu %6zu %8d %9.1f %9.1f %9llu %9llu %9llu %5zu %5zu\n",
                (unsigned long long)world.stats.steps, population, world.MaxGeneration(), neurons, synapses,
                (unsigned long long)world.stats.births, (unsigned long long)world.stats.spawned,
                (unsigned long long)world.stats.deaths, world.food.size(), world.airPockets.size());
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
    cfg.organismSensor = HasFlag(argc, argv, "--organism-sensor");
    cfg.positionSensor = HasFlag(argc, argv, "--position-sensor");
    cfg.slices = HasFlag(argc, argv, "--slices");

    World world(cfg);
    std::printf("Nworks headless run: %llu steps, seed %u\n", steps, rng::CurrentSeed());
    std::printf("%11s %6s %8s %9s %9s %9s %9s %9s %5s %5s\n", "step", "alive", "max gen", "neurons",
                "synapses", "births", "spawned", "deaths", "food", "air");

    const auto start = std::chrono::steady_clock::now();
    for (unsigned long long i = 1; i <= steps; ++i) {
        world.Step();
        if (i % report == 0 || i == steps) PrintRow(world);
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::printf("Done in %.1f s (%.0f steps/s). Neurons and synapses are per-organism averages.\n", seconds,
                seconds > 0.0 ? double(steps) / seconds : 0.0);
    return 0;
}
