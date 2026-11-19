#include "brain_test.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>
#include "brain_io.h"
#include "random.h"
#include "sensors.h"
#include "world.h"

namespace {

constexpr float kTwoPi = 6.28318530718f;

struct Options {
    int trials = 40;
    float distance = 120.0f;
    int copies = 20;
    int liveRuns = 3;
    float difficulty = 0.2f;
    uint32_t seed = 1;
    int maxChoiceSteps = 800;
    int maxLiveSteps = 60000;
    std::string saveBest;
    std::vector<std::string> files;
};

bool ParseOptions(int argc, char** argv, Options& opt) {
    auto number = [&](int& i, double& value) {
        if (i + 1 >= argc) {
            std::fprintf(stderr, "%s needs a number\n", argv[i]);
            return false;
        }
        try {
            value = std::stod(argv[++i]);
        } catch (const std::exception&) {
            std::fprintf(stderr, "%s needs a number, got '%s'\n", argv[i - 1], argv[i]);
            return false;
        }
        return true;
    };
    bool inFiles = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        double v = 0.0;
        if (arg == "--test") {
            inFiles = true;
        } else if (arg == "--trials") {
            if (!number(i, v)) return false;
            opt.trials = std::max(1, int(v));
        } else if (arg == "--distance") {
            if (!number(i, v)) return false;
            opt.distance = float(v);
        } else if (arg == "--copies") {
            if (!number(i, v)) return false;
            opt.copies = std::max(1, int(v));
        } else if (arg == "--live-runs") {
            if (!number(i, v)) return false;
            opt.liveRuns = std::max(0, int(v));
        } else if (arg == "--difficulty") {
            if (!number(i, v)) return false;
            opt.difficulty = std::clamp(float(v), 0.0f, 1.0f);
        } else if (arg == "--seed") {
            if (!number(i, v)) return false;
            opt.seed = std::max(1u, uint32_t(v));
        } else if (arg == "--save-best") {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "--save-best needs a directory\n");
                return false;
            }
            opt.saveBest = argv[++i];
        } else if (arg.rfind("--", 0) == 0) {
            std::fprintf(stderr, "unknown option %s\n", arg.c_str());
            return false;
        } else if (inFiles) {
            opt.files.push_back(arg);
        }
    }
    if (opt.files.empty()) {
        std::fprintf(stderr, "usage: --test FILE... [--trials N] [--distance D] [--copies N] [--difficulty D]\n");
        return false;
    }
    return true;
}

// Index of the first vitals input (hunger), or -1.
int VitalsInput(const Organism& o) {
    for (const auto& m : o.modules) {
        if (dynamic_cast<const VitalsSensor*>(m.get())) return m->firstInput;
    }
    return -1;
}

// Vitals input values for a well-fed organism with full air (see VitalsSensor).
std::vector<std::pair<int, float>> FrozenVitals(const Organism& o) {
    const int first = VitalsInput(o);
    if (first < 0) return {};
    return {{first, 0.3f}, {first + 1, 0.5f}};
}

bool Touching(const Organism& o, float x, float y, float radius) {
    const float dx = x - o.x, dy = y - o.y, reach = o.radius + radius;
    return dx * dx + dy * dy < reach * reach;
}

enum class Pick { Food, Air, Neither };

struct ChoiceTally {
    int food = 0, air = 0, neither = 0;
    void Add(Pick p) { (p == Pick::Food ? food : p == Pick::Air ? air : neither)++; }
    int Total() const { return food + air + neither; }
    double FoodShare() const { return Total() ? double(food) / Total() : 0.0; }
    double AirShare() const { return Total() ? double(air) / Total() : 0.0; }
    double NeitherShare() const { return Total() ? double(neither) / Total() : 0.0; }
};

// One choice trial: food at `angle`, a spring opposite, both `distance` away.
Pick ChoiceTrial(const SavedOrganism& saved, const Config& cfg, const Options& opt, float angle, uint8_t foodKind,
                 float healthShare, float airShare) {
    World world(cfg);
    world.food.clear();
    world.bigFood.clear();
    world.airPockets.clear();
    world.springs.clear();
    world.predators.clear();
    world.difficulty = 0.0f;

    const float cx = cfg.worldWidth * 0.5f, cy = cfg.worldHeight * 0.5f;
    const float fx = cx + std::cos(angle) * opt.distance, fy = cy + std::sin(angle) * opt.distance;
    const float sx = cx - std::cos(angle) * opt.distance, sy = cy - std::sin(angle) * opt.distance;
    world.food.push_back(Resource{fx, fy, cfg.foodRadius, 0, foodKind});
    world.springs.push_back(Spring{sx, sy, cfg.springRadius, cfg.springCapacity});

    Organism o(1, cx, cy, cfg);
    std::string error;
    if (!MakeOrganism(saved, world, cx, cy, o, error)) return Pick::Neither;
    o.id = 1;
    o.health = healthShare * cfg.maxHealth;
    o.air = airShare * cfg.maxAir;

    for (int step = 0; step < opt.maxChoiceSteps; ++step) {
        o.Update(world);
        const bool food = Touching(o, fx, fy, cfg.foodRadius);
        const bool air = Touching(o, sx, sy, cfg.springRadius);
        if (food) return Pick::Food;  // they're 2 * distance apart, so never both
        if (air) return Pick::Air;
        if (o.health <= 0.0f || o.air <= 0.0f) break;
    }
    return Pick::Neither;
}

struct ChoiceResult {
    ChoiceTally hungry, breathless;
    // How much being hungry instead of breathless raises the chance of picking
    // food, in percentage points. 0 = the vitals inputs make no difference.
    double Switch() const { return 100.0 * (hungry.FoodShare() - breathless.FoodShare()); }
    bool Passed() const { return hungry.FoodShare() >= 0.6 && breathless.AirShare() >= 0.6; }
};

ChoiceResult ChoiceTest(const SavedOrganism& saved, const Config& cfg, const Options& opt) {
    rng::Seed(opt.seed);
    std::vector<float> angles;
    for (int i = 0; i < opt.trials; ++i) angles.push_back(kTwoPi * (float(i) + rng::Float(0.0f, 1.0f)) / float(opt.trials));
    ChoiceResult r;
    for (int i = 0; i < opt.trials; ++i) {
        const uint8_t kind = uint8_t(i % 2);  // half the trials use each food colour
        r.hungry.Add(ChoiceTrial(saved, cfg, opt, angles[size_t(i)], kind, 0.2f, 1.0f));
        r.breathless.Add(ChoiceTrial(saved, cfg, opt, angles[size_t(i)], kind, 1.0f, 0.35f));
    }
    return r;
}

// Mean lifespan (steps) of `copies` clones in a live world.
double LiveLifespan(const SavedOrganism& saved, Config cfg, const Options& opt, uint32_t seed, bool frozen) {
    cfg.seed = seed;
    cfg.minPopulation = 0;
    cfg.foodToReproduce = 1 << 30;
    cfg.curriculumWindow = 0;
    World world(cfg);
    world.difficulty = opt.difficulty;
    for (int i = 0; i < opt.copies; ++i) {
        const float x = rng::Float(cfg.spawnMinX, cfg.spawnMaxX), y = rng::Float(cfg.spawnMinY, cfg.spawnMaxY);
        Organism o(uint64_t(i + 1), x, y, cfg);
        std::string error;
        if (!MakeOrganism(saved, world, x, y, o, error)) return 0.0;
        o.id = uint64_t(i + 1);
        if (frozen) o.clampedInputs = FrozenVitals(o);
        world.organisms.push_back(std::move(o));
    }
    double lived = 0.0;
    for (int step = 0; step < opt.maxLiveSteps && !world.organisms.empty(); ++step) {
        world.Step();
        lived += double(world.organisms.size());
    }
    return lived / double(opt.copies);
}

std::string Percent(double share) { return std::to_string(int(std::lround(100.0 * share))) + "%"; }

}  // namespace

int RunBrainTest(int argc, char** argv) {
    Options opt;
    if (!ParseOptions(argc, argv, opt)) return 1;
    if (!opt.saveBest.empty()) std::filesystem::create_directories(opt.saveBest);

    std::printf("Choice test: %d angles, food and spring %.0f px away in opposite directions.\n", opt.trials,
                opt.distance);
    std::printf("  hungry = 20%% health, full air; breathless = full health, 35%% air.\n");
    std::printf("  switch = food%% when hungry - food%% when breathless (0 = hunger/air inputs make no difference)\n");
    if (opt.liveRuns > 0) {
        std::printf("Live test: %d copies x %d runs at %.0f%% difficulty; mean lifespan in steps, as they are vs "
                    "with hunger/air inputs frozen at 'full'.\n",
                    opt.copies, opt.liveRuns, opt.difficulty * 100.0f);
    }
    std::printf("\n%-40s %5s %7s | %-17s | %-17s | %6s | %8s %8s %6s\n", "organism", "gen", "neurons",
                "hungry food/air/-", "breath food/air/-", "switch", "life", "frozen", "ratio");

    int passed = 0;
    for (const std::string& path : opt.files) {
        Config cfg;
        SavedOrganism saved;
        std::string error;
        if (!LoadOrganism(path, cfg, saved, error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            continue;
        }
        {
            World probe(cfg);
            Organism o(1, 0.0f, 0.0f, cfg);
            if (!MakeOrganism(saved, probe, 0.0f, 0.0f, o, error)) {
                std::fprintf(stderr, "%s: %s\n", path.c_str(), error.c_str());
                continue;
            }
        }

        const ChoiceResult choice = ChoiceTest(saved, cfg, opt);
        double life = 0.0, frozen = 0.0;
        for (int run = 0; run < opt.liveRuns; ++run) {
            life += LiveLifespan(saved, cfg, opt, opt.seed + 1000u + uint32_t(run), false);
            frozen += LiveLifespan(saved, cfg, opt, opt.seed + 1000u + uint32_t(run), true);
        }
        if (opt.liveRuns > 0) {
            life /= opt.liveRuns;
            frozen /= opt.liveRuns;
        }

        const std::string name = std::filesystem::path(path).filename().string();
        auto tally = [](const ChoiceTally& t) {
            return Percent(t.FoodShare()) + "/" + Percent(t.AirShare()) + "/" + Percent(t.NeitherShare());
        };
        std::printf("%-40s %5d %7zu | %-17s | %-17s | %+5.0f  | %8.0f %8.0f %6.2f%s\n", name.c_str(),
                    saved.generation, saved.brain.NeuronCount(), tally(choice.hungry).c_str(),
                    tally(choice.breathless).c_str(), choice.Switch(), life, frozen,
                    frozen > 0.0 ? life / frozen : 0.0, choice.Passed() ? "  PASS" : "");
        std::fflush(stdout);

        if (choice.Passed()) {
            ++passed;
            if (!opt.saveBest.empty()) {
                char note[256];
                std::snprintf(note, sizeof(note),
                              "choice test: hungry picks food %s, breathless picks air %s, switch %+.0f; "
                              "live lifespan %.0f vs %.0f with vitals frozen",
                              Percent(choice.hungry.FoodShare()).c_str(),
                              Percent(choice.breathless.AirShare()).c_str(), choice.Switch(), life, frozen);
                World world(cfg);
                Organism o(saved.id, 0.0f, 0.0f, cfg);
                MakeOrganism(saved, world, 0.0f, 0.0f, o, error);
                o.id = saved.id;
                SaveOrganism(o, cfg, (std::filesystem::path(opt.saveBest) / name).string(), note);
            }
        }
    }
    std::printf("\n%d of %zu passed (hungry picks food >= 60%% and breathless picks air >= 60%%).\n", passed,
                opt.files.size());
    return 0;
}
