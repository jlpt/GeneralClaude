// Evolves a world and scores its creatures on the intelligence test battery.
//
//   nworks_iq SEED CHECKPOINT[,CHECKPOINT...] [TRIALS] [key=value ...]
//
// At the start it scores 40 brand-new random brains; at each checkpoint it
// scores a copy of every creature alive (TRIALS trials per creature per test,
// default 10). Each test reports its success rate; Flee and FleeControl also
// report the mean heading away from the predator (+1 = straight away).
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "common.h"
#include "random.h"
#include "scenarios.h"

namespace {

struct TestScore {
    int n = 0, success = 0, bit = 0;
    double score = 0.0;
};

void RunBattery(const Config& cfg, std::vector<Organism>& group, int trials, const char* label) {
    TestScore scores[int(Test::Count)];
    for (int t = 0; t < int(Test::Count); ++t) {
        int k = 0;
        for (Organism& src : group) {
            for (int trial = 0; trial < trials; ++trial, ++k) {
                Scenario s(Test(t), cfg, uint32_t((t + 1) * 100000 + k), CloneOf(src, 1, cfg));
                while (!s.Done()) s.Step();
                TestScore& ts = scores[t];
                ++ts.n;
                ts.success += s.Success() ? 1 : 0;
                ts.bit += s.Bit() ? 1 : 0;
                ts.score += s.Score();
            }
        }
    }
    std::printf("  %s (%zu brains x %d trials):\n", label, group.size(), trials);
    for (int t = 0; t < int(Test::Count); ++t) {
        const TestScore& ts = scores[t];
        std::printf("    %-15s success %5.1f%%", TestName(Test(t)), ts.n ? 100.0 * ts.success / ts.n : 0.0);
        if (Test(t) == Test::Flee || Test(t) == Test::FleeControl) std::printf("  heading %+.2f", ts.n ? ts.score / ts.n : 0.0);
        if (Test(t) == Test::PoisonChoice)
            std::printf("  (ate one in %.1f%%; picked purple in %.1f%% of those)", ts.n ? 100.0 * ts.bit / ts.n : 0.0,
                        ts.bit ? 100.0 * ts.success / ts.bit : 0.0);
        std::printf("\n");
    }
    std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s SEED CHECKPOINT[,CHECKPOINT...] [TRIALS] [key=value ...]\n", argv[0]);
        return 1;
    }
    Config cfg;
    cfg.seed = uint32_t(std::strtoul(argv[1], nullptr, 10));
    std::vector<unsigned long long> checkpoints;
    for (char* p = argv[2]; *p;) {
        checkpoints.push_back(std::strtoull(p, &p, 10));
        if (*p == ',') ++p;
    }
    int trials = 10, first = 3;
    if (argc > 3 && std::string(argv[3]).find('=') == std::string::npos) { trials = std::atoi(argv[3]); first = 4; }
    std::string settings;
    for (int i = first; i < argc; ++i) settings += std::string(" ") + argv[i];
    if (!ApplyOverrides(cfg, settings)) { std::fprintf(stderr, "unknown setting in:%s\n", settings.c_str()); return 1; }
    std::printf("seed %u, settings:%s\n", cfg.seed, settings.empty() ? " (defaults)" : settings.c_str());

    World world(cfg);
    const std::mt19937 afterSeed = rng::Engine();
    {
        Config rc = cfg;
        rc.seed = cfg.seed + 777;
        rc.refillFromSurvivors = false;
        rc.minPopulation = 40;
        World scratch(rc);
        while (scratch.organisms.size() < 40) scratch.Step();
        std::vector<Organism> randomBrains;
        for (Organism& o : scratch.organisms) randomBrains.push_back(CloneOf(o, 1, rc));
        RunBattery(cfg, randomBrains, trials, "random brains");
    }
    rng::Engine() = afterSeed;

    unsigned long long done = 0;
    for (unsigned long long checkpoint : checkpoints) {
        for (; done < checkpoint; ++done) {
            world.Step();
            if ((done + 1) % 1000000 == 0) {
                const size_t n = world.organisms.size();
                std::printf("  step %9llu | alive %3zu | max gen %5d | diff %3.0f%% | neurons %6.1f | good bites %llu, "
                            "poison bites %llu | eaten by predators %llu\n",
                            done + 1, n, world.MaxGeneration(), world.difficulty * 100.0f,
                            n ? double(world.TotalNeurons()) / double(n) : 0.0,
                            (unsigned long long)world.stats.goodBites, (unsigned long long)world.stats.poisonBites,
                            (unsigned long long)world.stats.eatenByPredators);
                std::fflush(stdout);
            }
        }
        const std::mt19937 saved = rng::Engine();
        std::vector<Organism> evolved;
        for (Organism& o : world.organisms) evolved.push_back(CloneOf(o, 1, world.cfg));
        char label[64];
        std::snprintf(label, sizeof(label), "evolved, step %llu", checkpoint);
        RunBattery(cfg, evolved, trials, label);
        rng::Engine() = saved;
    }
    return 0;
}
