// Nworks experiment runner.
//
//   exp SEED STEPS REPORT [tests=STEP,STEP,...] [key=value ...]
//
// Evolves a world with the given Config overrides and prints a row every
// REPORT steps. At each test step it copies every creature and runs a
// within-life learning test (see LearningTest).
//
// build: the nworks_exp target in CMakeLists.txt
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>
#include "actuators.h"
#include "organism.h"
#include "sensors.h"
#include "random.h"
#include "world.h"

namespace {

constexpr float kTwoPi = 6.28318530718f;
constexpr float kCx = 650.0f, kCy = 384.0f;

// --- Config overrides ---------------------------------------------------------
bool ApplyOverride(Config& c, const std::string& key, const std::string& value) {
    std::map<std::string, float*> floats = {
        {"maxHealth", &c.maxHealth}, {"maxAir", &c.maxAir}, {"moveSpeed", &c.moveSpeed},
        {"brainCostPerNeuron", &c.brainCostPerNeuron}, {"brainCostPerSynapse", &c.brainCostPerSynapse},
        {"foodSpawnChance", &c.foodSpawnChance}, {"poisonDamage", &c.poisonDamage},
        {"traceDecay", &c.traceDecay}, {"learnedForgetting", &c.learnedForgetting}, {"rewardDecay", &c.rewardDecay},
        {"maxInitialLearningRate", &c.maxInitialLearningRate}, {"learningRateStep", &c.learningRateStep},
        {"learningRateChance", &c.learningRateChance}, {"mutationFraction", &c.mutationFraction},
        {"worldWidth", &c.worldWidth}, {"worldHeight", &c.worldHeight}, {"visionRange", &c.visionRange},
        {"predatorBaseSpeed", &c.predatorBaseSpeed}, {"predatorExtraSpeed", &c.predatorExtraSpeed},
        {"signalCost", &c.signalCost}, {"randomRefillShare", &c.randomRefillShare}, {"poisonShare", &c.poisonShare}, {"springRecharge", &c.springRecharge}, {"curriculumStep", &c.curriculumStep},
    };
    std::map<std::string, int*> ints = {
        {"minPopulation", &c.minPopulation}, {"maxAge", &c.maxAge}, {"minFood", &c.minFood}, {"foodToReproduce", &c.foodToReproduce}, {"maxFood", &c.maxFood},
        {"initialFood", &c.initialFood}, {"poisonSwapInterval", &c.poisonSwapInterval},
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
        {"eatAction", &c.eatAction}, {"refillFromSurvivors", &c.refillFromSurvivors},
    };
    if (auto it = floats.find(key); it != floats.end()) { *it->second = std::stof(value); return true; }
    if (auto it = ints.find(key); it != ints.end()) { *it->second = std::stoi(value); return true; }
    if (auto it = bools.find(key); it != bools.end()) { *it->second = value == "1" || value == "true" || value == "on"; return true; }
    return false;
}

// --- Copies and arenas ------------------------------------------------------
Organism CloneOf(Organism& src, uint64_t id, const Config& cfg) {
    const float saved = src.mutationRate;
    src.mutationRate = 0.0f;
    Organism c = src.MakeOffspring(id, cfg);
    src.mutationRate = saved;
    c.generation = src.generation;
    return c;
}

// An empty arena where colour `poisonKind` is poisonous (no damage; the bad
// taste and the learning signal still happen).
World MakeArena(const Config& base, uint32_t seed, int poisonKind, bool plastic) {
    Config c = base;
    c.seed = seed;
    c.minPopulation = 0;
    c.foodSpawnChance = 0.0f;
    c.bigFoodSpawnChance = 0.0f;
    c.airSpawnChance = 0.0f;
    c.curriculumWindow = 0;
    c.poison = true;
    c.poisonDamage = 0.0f;
    c.poisonSwapInterval = 1000000000;
    c.plasticity = plastic;
    World w(c);
    w.food.clear();
    w.bigFood.clear();
    w.springs.clear();
    w.airPockets.clear();
    w.predators.clear();
    w.difficulty = 0.0f;
    if (poisonKind == 1) w.stats.steps = 1000000000ULL;  // PoisonKind() = (steps / interval) % 2
    return w;
}


// --- Within-life learning test -------------------------------------------------
// One life, 12 rounds. Each round: the creature is put in the middle, hungry,
// with one food of each colour 120 px away on opposite sides. The round ends
// at the first bite (or after 400 steps), then 60 quiet steps let the reward
// signal fade before the next round. Learning shows up as more safe picks in
// later rounds than in early ones, and more than the same brain frozen.
constexpr int kRounds = 12;

struct LearnTally {
    int bites[3] = {0, 0, 0}, safe[3] = {0, 0, 0};  // rounds 1-4, 5-8, 9-12
    int rounds = 0;
    void Add(int round, bool bit, bool wasSafe) {
        ++rounds;
        if (!bit) return;
        ++bites[round / 4];
        safe[round / 4] += wasSafe ? 1 : 0;
    }
    double SafePct(int block) const { return bites[block] ? 100.0 * safe[block] / bites[block] : -1.0; }
    double BitePct() const {
        return rounds ? 100.0 * (bites[0] + bites[1] + bites[2]) / rounds : 0.0;
    }
};

void OneLife(const Config& base, Organism& src, uint32_t seed, int poisonKind, bool plastic, float forcedRate,
             LearnTally& tally) {
    World w = MakeArena(base, seed, poisonKind, plastic);
    Organism o = CloneOf(src, 1, w.cfg);
    if (forcedRate >= 0.0f) o.brain.SetLearningRate(forcedRate);
    o.health = w.cfg.maxHealth;
    o.air = w.cfg.maxAir;
    o.reward = 0.0f;
    w.organisms.push_back(std::move(o));
    std::mt19937 g(seed * 2654435761u + 13u);
    std::uniform_real_distribution<float> angle(0.0f, kTwoPi);
    for (int round = 0; round < kRounds; ++round) {
        if (w.organisms.empty()) { tally.Add(round, false, false); continue; }
        Organism& c = w.organisms[0];
        c.x = kCx;
        c.y = kCy;
        c.health = 0.25f * w.cfg.maxHealth;
        c.air = w.cfg.maxAir;
        c.foodEaten = 0;  // no reproduction in the arena
        const float a = angle(g);
        w.food.clear();
        // Which colour goes on which side alternates, so side habits cancel out.
        const uint8_t first = uint8_t(round % 2);
        w.food.push_back(Resource{kCx + 120.0f * std::cos(a), kCy + 120.0f * std::sin(a), w.cfg.foodRadius, w.stats.steps, first});
        w.food.push_back(Resource{kCx - 120.0f * std::cos(a), kCy - 120.0f * std::sin(a), w.cfg.foodRadius, w.stats.steps, uint8_t(1 - first)});
        const uint64_t poisonBefore = w.stats.poisonBites, goodBefore = w.stats.goodBites;
        for (int t = 0; t < 400 && w.food.size() == 2 && !w.organisms.empty(); ++t) w.Step();
        const bool bit = w.stats.poisonBites + w.stats.goodBites > poisonBefore + goodBefore;
        const bool wasSafe = w.stats.goodBites > goodBefore && w.stats.poisonBites == poisonBefore;
        tally.Add(round, bit, wasSafe);
        w.food.clear();
        for (int t = 0; t < 60 && !w.organisms.empty(); ++t) w.Step();  // let the reward fade
    }
}

void LearningTest(const Config& cfg, std::vector<Organism>& group, const char* label) {
    LearnTally plastic, frozen, boosted;
    uint32_t seed = 1;
    for (Organism& src : group) {
        for (int rep = 0; rep < 2; ++rep) {
            for (int poisonKind = 0; poisonKind < 2; ++poisonKind, ++seed) {
                OneLife(cfg, src, seed, poisonKind, true, -1.0f, plastic);
                OneLife(cfg, src, seed, poisonKind, false, -1.0f, frozen);
                OneLife(cfg, src, seed, poisonKind, true, 0.1f, boosted);
            }
        }
    }
    auto line = [](const char* name, const LearnTally& t) {
        std::printf("      %-22s bit in %5.1f%% of rounds | safe picks: rounds 1-4 %5.1f%%  5-8 %5.1f%%  9-12 %5.1f%%"
                    "  (bites %d/%d/%d)\n",
                    name, t.BitePct(), t.SafePct(0), t.SafePct(1), t.SafePct(2), t.bites[0], t.bites[1], t.bites[2]);
    };
    std::printf("  learning test, %s (%zu creatures, 4 lives each, %d rounds per life):\n", label, group.size(), kRounds);
    line("own learning rate", plastic);
    line("learning frozen", frozen);
    line("learning rate forced 0.1", boosted);
    std::fflush(stdout);
}

// --- Rows -------------------------------------------------------------------------
struct RowState {
    uint64_t poison = 0, good = 0, births = 0;
    uint64_t phasePoison[5] = {}, phaseGood[5] = {};
    // Poison share by how many bites this creature already took in the current poison period.
    uint64_t nthPoison[5] = {}, nthGood[5] = {};  // bins: 1st, 2nd, 3rd-4th, 5th-8th, 9th+
};

struct BiteTracker {
    struct Entry { uint64_t period = ~0ULL; int bites = 0; };
    std::map<uint64_t, Entry> byId;
    std::map<uint64_t, std::pair<int, int>> before;  // id -> (good, poison) bites so far

    void Before(const World& w) {
        before.clear();
        for (const Organism& o : w.organisms) before[o.id] = {o.goodBitesInLife, o.poisonBitesInLife};
    }
    static int Bin(int n) { return n == 0 ? 0 : n == 1 ? 1 : n < 4 ? 2 : n < 8 ? 3 : 4; }
    void After(const World& w, RowState& st) {
        if (!w.cfg.poison || w.cfg.poisonSwapInterval <= 0) return;
        const uint64_t period = (w.stats.steps - 1) / uint64_t(w.cfg.poisonSwapInterval);
        for (const Organism& o : w.organisms) {
            auto it = before.find(o.id);
            const int g0 = it == before.end() ? 0 : it->second.first, p0 = it == before.end() ? 0 : it->second.second;
            const int good = o.goodBitesInLife - g0, poison = o.poisonBitesInLife - p0;
            for (int k = 0; k < good + poison; ++k) {
                Entry& e = byId[o.id];
                if (e.period != period) { e.period = period; e.bites = 0; }
                const int b = Bin(e.bites++);
                if (k < poison) ++st.nthPoison[b]; else ++st.nthGood[b];
            }
        }
        if (byId.size() > 200000) byId.clear();
    }
};

void PrintRow(const World& w, RowState& st) {
    const size_t n = w.organisms.size();
    double learn = 0.0, learners = 0.0;
    for (const Organism& o : w.organisms) {
        learn += o.brain.LearningRate();
        learners += o.brain.LearningRate() > 0.005f ? 1.0 : 0.0;
    }
    const uint64_t p = w.stats.poisonBites - st.poison, g = w.stats.goodBites - st.good;
    size_t poisonItems = 0;
    for (const Resource& r : w.food) poisonItems += (w.cfg.poison && r.kind == w.PoisonKind()) ? 1 : 0;
    std::printf("%10llu | food %3d (%2.0f%% bad) | alive %3zu gen %5d diff %3.0f%% | neurons %6.1f | poison %5.1f%% | by fifth of swap period:",
                (unsigned long long)w.stats.steps, w.cfg.maxFood,
                w.food.empty() ? 0.0 : 100.0 * double(poisonItems) / double(w.food.size()), n, w.MaxGeneration(), w.difficulty * 100.0f,
                n ? double(w.TotalNeurons()) / double(n) : 0.0, p + g ? 100.0 * double(p) / double(p + g) : 0.0);
    for (int b = 0; b < 5; ++b) {
        const uint64_t bp = st.phasePoison[b], bg = st.phaseGood[b];
        std::printf(" %5.1f", bp + bg ? 100.0 * double(bp) / double(bp + bg) : 0.0);
        st.phasePoison[b] = st.phaseGood[b] = 0;
    }
    std::printf(" | by bite# in period:");
    for (int b = 0; b < 5; ++b) {
        const uint64_t bp = st.nthPoison[b], bg = st.nthGood[b];
        std::printf(" %5.1f", bp + bg ? 100.0 * double(bp) / double(bp + bg) : -1.0);
    }
    std::printf(" (n");
    for (int b = 0; b < 5; ++b) std::printf(" %llu", (unsigned long long)(st.nthPoison[b] + st.nthGood[b]));
    std::printf(")");
    std::printf(" | learn %.4f (%.0f%% >0.005) | bites %llu births %llu spawned %llu eaten %llu\n", n ? learn / double(n) : 0.0,
                n ? 100.0 * learners / double(n) : 0.0, (unsigned long long)(p + g),
                (unsigned long long)(w.stats.births - st.births), (unsigned long long)w.stats.spawned,
                (unsigned long long)w.stats.eatenByPredators);
    st.poison = w.stats.poisonBites;
    st.good = w.stats.goodBites;
    st.births = w.stats.births;
    std::fflush(stdout);
}


// --- Hand-built circuits (capacity test) ----------------------------------------
// Builds a brain by hand on a creature's body and measures whether the
// plasticity rule, at various learning rates, makes it learn which colour is
// poisonous within one life.
struct Layout {
    int foodA = -1, foodB = -1, sectors = 8, hunger = -1, smellA = -1, smellB = -1, right = -1;
    int air = -1, bigFood = -1, predator = -1, breath = -1;
};

Layout FindLayout(const Organism& o, const Config& cfg) {
    Layout L;
    L.sectors = cfg.visionSectors;
    for (const auto& m : o.modules) {
        if (dynamic_cast<const VisionSensor*>(m.get())) {
            // Same slot order as VisionSensor: FoodA, FoodB, Air, Creature, BigFood, Predator (unused kinds skipped).
            const bool used[6] = {true, cfg.poison, true, true, cfg.cooperativeFood, cfg.predators};
            int slot[6], next = 0;
            for (int kind = 0; kind < 6; ++kind) slot[kind] = used[kind] ? next++ : -1;
            auto at = [&](int kind) { return slot[kind] < 0 ? -1 : m->firstInput + slot[kind] * cfg.visionSectors; };
            L.foodA = at(0);
            L.foodB = at(1);
            L.air = at(2);
            L.bigFood = at(4);
            L.predator = at(5);
        }
        if (dynamic_cast<const VitalsSensor*>(m.get())) { L.hunger = m->firstInput; L.breath = m->firstInput + 1; }
        if (dynamic_cast<const ColourSmellSensor*>(m.get())) { L.smellA = m->firstInput; L.smellB = m->firstInput + 1; }
        if (dynamic_cast<const MovementModule*>(m.get())) L.right = o.brain.NumInputs() + m->firstOutput;
    }
    return L;
}

// Wires `from` to the four movement outputs so that it pushes toward angle theta.
void ConnectToward(NeuralNetwork& nn, const Layout& L, int from, float theta, float w) {
    const float c = std::cos(theta), s = std::sin(theta);
    if (c > 0.05f) nn.Connect(from, L.right + 0, w * c);   // right
    if (c < -0.05f) nn.Connect(from, L.right + 1, -w * c); // left
    if (s > 0.05f) nn.Connect(from, L.right + 2, w * s);   // down
    if (s < -0.05f) nn.Connect(from, L.right + 3, -w * s); // up
}

float SectorAngle(int k, int sectors) { return -3.14159265f + (float(k) + 0.5f) * kTwoPi / float(sectors); }

struct HandParams {
    std::string circuit = "direct";
    float wMotor = 0.1f, wFood = 0.3f, wGate = 0.3f, wValue = 0.5f, wSmell = 0.3f, threshold = 1.0f, jitter = 0.2f;
    // "full" circuit
    float wNear = 0.3f, wNearMotor = 0.3f, wFar = 0.0f, wInhibit = 0.5f;
    float wBreath = 0.06f, wMigrate = 0.3f, wAirSeen = 0.08f, wAirGate = 0.05f, wAirMotor = 0.5f;
    float wFlee = 0.5f, wBigFood = 0.15f;
};

NeuralNetwork BuildCircuit(const Organism& tmpl, const Config& cfg, const HandParams& hp, uint32_t seed) {
    const Layout L = FindLayout(tmpl, cfg);
    NeuralNetwork nn(tmpl.brain.NumInputs(), tmpl.brain.NumOutputs(), 0, 0, cfg);
    std::mt19937 g(seed);
    std::uniform_real_distribution<float> jit(1.0f - hp.jitter, 1.0f + hp.jitter);  // innate random biases
    const int food[2] = {L.foodA, L.foodB};
    if (hp.circuit == "direct") {
        // Food of either colour in sector k pulls the creature toward sector k.
        for (int c = 0; c < 2; ++c)
            for (int k = 0; k < L.sectors; ++k)
                ConnectToward(nn, L, food[c] + k, SectorAngle(k, L.sectors), hp.wMotor * jit(g));
    } else if (hp.circuit == "full") {
        // Food: per colour and sector, a weak direct pull plus a "near food" relay
        // that only fires for close food; relays inhibit each other so the
        // nearest food wins instead of all pulls cancelling out.
        std::vector<int> relays;
        for (int c = 0; c < 2; ++c) {
            if (food[c] < 0) continue;
            for (int k = 0; k < L.sectors; ++k) {
                const float theta = SectorAngle(k, L.sectors);
                if (hp.wFar > 0.0f) ConnectToward(nn, L, food[c] + k, theta, hp.wFar * jit(g));
                const int relay = nn.AddHiddenNeuron(hp.threshold, 1);
                nn.Connect(food[c] + k, relay, hp.wNear * jit(g));
                ConnectToward(nn, L, relay, theta, hp.wNearMotor);
                relays.push_back(relay);
            }
        }
        for (int a : relays)
            for (int b : relays)
                if (a != b) nn.Connect(a, b, -hp.wInhibit);
        // Air: when breathless, a "migrate" neuron pushes up-left (toward the
        // springs from every food patch) and opens per-sector spring relays.
        if (L.air >= 0) {
            const int migrate = nn.AddHiddenNeuron(hp.threshold, 1);
            nn.Connect(L.breath, migrate, hp.wBreath);
            nn.Connect(migrate, L.right + 3, hp.wMigrate);  // up
            nn.Connect(migrate, L.right + 1, hp.wMigrate);  // left
            for (int k = 0; k < L.sectors; ++k) {
                const int relay = nn.AddHiddenNeuron(hp.threshold, 1);
                nn.Connect(L.air + k, relay, hp.wAirSeen);
                nn.Connect(migrate, relay, hp.wAirGate);
                ConnectToward(nn, L, relay, SectorAngle(k, L.sectors), hp.wAirMotor);
            }
        }
        // Predators: run directly away.
        if (L.predator >= 0)
            for (int k = 0; k < L.sectors; ++k)
                ConnectToward(nn, L, L.predator + k, SectorAngle(k, L.sectors) + 3.14159265f, hp.wFlee);
        // Big food: approach it.
        if (L.bigFood >= 0)
            for (int k = 0; k < L.sectors; ++k)
                ConnectToward(nn, L, L.bigFood + k, SectorAngle(k, L.sectors), hp.wBigFood);
    } else {
        // Gated: one "value" neuron per colour (driven by hunger and that colour's smell)
        // must be active for that colour's per-sector relay neurons to fire.
        const int smell[2] = {L.smellA, L.smellB};
        for (int c = 0; c < 2; ++c) {
            const int value = nn.AddHiddenNeuron(hp.threshold, 1);
            nn.Connect(L.hunger, value, hp.wValue * jit(g));
            nn.Connect(smell[c], value, hp.wSmell);
            for (int k = 0; k < L.sectors; ++k) {
                const int relay = nn.AddHiddenNeuron(hp.threshold, 1);
                nn.Connect(food[c] + k, relay, hp.wFood);
                nn.Connect(value, relay, hp.wGate);
                ConnectToward(nn, L, relay, SectorAngle(k, L.sectors), hp.wMotor);
            }
        }
    }
    return nn;
}

int HandBuilt(int argc, char** argv) {
    Config cfg;
    cfg.predators = false; cfg.signals = false; cfg.cooperativeFood = false; cfg.colourSmell = true;
    HandParams hp;
    std::vector<float> rates = {0.0f, 0.02f, 0.05f, 0.1f, 0.2f, 0.5f};
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        const size_t eq = arg.find('=');
        const std::string key = arg.substr(0, eq), value = arg.substr(eq + 1);
        if (key == "circuit") hp.circuit = value;
        else if (key == "wMotor") hp.wMotor = std::stof(value);
        else if (key == "wFood") hp.wFood = std::stof(value);
        else if (key == "wGate") hp.wGate = std::stof(value);
        else if (key == "wValue") hp.wValue = std::stof(value);
        else if (key == "wSmell") hp.wSmell = std::stof(value);
        else if (key == "threshold") hp.threshold = std::stof(value);
        else if (key == "jitter") hp.jitter = std::stof(value);
        else if (key == "wNear") hp.wNear = std::stof(value);
        else if (key == "wNearMotor") hp.wNearMotor = std::stof(value);
        else if (key == "wFar") hp.wFar = std::stof(value);
        else if (key == "wInhibit") hp.wInhibit = std::stof(value);
        else if (key == "sectors") cfg.visionSectors = std::stoi(value);
        else if (!ApplyOverride(cfg, key, value)) { std::fprintf(stderr, "unknown %s\n", key.c_str()); return 1; }
    }
    // A creature body with the right modules.
    Config scratchCfg = cfg;
    scratchCfg.seed = 99;
    scratchCfg.minPopulation = 1;
    World scratch(scratchCfg);
    scratch.Step();
    Organism tmpl = CloneOf(scratch.organisms[0], 1, cfg);
    std::printf("hand-built '%s' circuit, %d sectors (wMotor %.2f wFood %.2f wGate %.2f wValue %.2f wSmell %.2f threshold %.2f)\n",
                hp.circuit.c_str(), cfg.visionSectors, hp.wMotor, hp.wFood, hp.wGate, hp.wValue, hp.wSmell, hp.threshold);
    for (float rate : rates) {
        LearnTally t;
        for (uint32_t s = 1; s <= 40; ++s) {
            tmpl.brain = BuildCircuit(tmpl, cfg, hp, s);  // a different innate bias per life
            for (int poisonKind = 0; poisonKind < 2; ++poisonKind)
                OneLife(cfg, tmpl, s * 2 + uint32_t(poisonKind), poisonKind, rate > 0.0f, rate, t);
        }
        std::printf("  learning rate %.2f: bit in %5.1f%% of rounds | safe picks: rounds 1-4 %5.1f%%  5-8 %5.1f%%  9-12 %5.1f%%"
                    "  (bites %d/%d/%d)\n", rate, t.BitePct(), t.SafePct(0), t.SafePct(1), t.SafePct(2), t.bites[0], t.bites[1],
                    t.bites[2]);
        std::fflush(stdout);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "handbuilt") return HandBuilt(argc, argv);
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s SEED STEPS REPORT [tests=A,B] [key=value ...]\n", argv[0]);
        return 1;
    }
    Config cfg;
    cfg.seed = uint32_t(std::strtoul(argv[1], nullptr, 10));
    const unsigned long long steps = std::strtoull(argv[2], nullptr, 10);
    const unsigned long long report = std::strtoull(argv[3], nullptr, 10);
    std::vector<unsigned long long> tests;
    std::string settings, seedCircuit;
    float seedRate = 0.0f;
    int seedCount = 30;
    bool seedMix = false;  // every other founder gets learning rate 0
    float seedWMotor = 0.1f;
    // Food curriculum: every foodWindow steps, thin the food (x0.9) if the
    // population needed no refills, thicken it (+5) if it needed 10 or more.
    float startDifficulty = 0.0f;
    bool foodCurriculum = false;
    int foodMin = 10;
    unsigned long long foodWindow = 20000;
    for (int i = 4; i < argc; ++i) {
        std::string arg = argv[i];
        const size_t eq = arg.find('=');
        if (eq == std::string::npos) { std::fprintf(stderr, "bad argument %s\n", argv[i]); return 1; }
        const std::string key = arg.substr(0, eq), value = arg.substr(eq + 1);
        if (key == "seed_circuit") { seedCircuit = value; settings += " " + arg; continue; }
        if (key == "seed_rate") { seedRate = std::stof(value); settings += " " + arg; continue; }
        if (key == "seed_count") { seedCount = std::stoi(value); settings += " " + arg; continue; }
        if (key == "start_difficulty") { startDifficulty = std::stof(value); settings += " " + arg; continue; }
        if (key == "food_curriculum") { foodCurriculum = value == "1"; settings += " " + arg; continue; }
        if (key == "food_min") { foodMin = std::stoi(value); settings += " " + arg; continue; }
        if (key == "food_window") { foodWindow = std::stoull(value); settings += " " + arg; continue; }
        if (key == "seed_wmotor") { seedWMotor = std::stof(value); settings += " " + arg; continue; }
        if (key == "seed_mix") { seedMix = value == "1"; settings += " " + arg; continue; }
        if (key == "seed_sectors") { cfg.visionSectors = std::stoi(value); settings += " " + arg; continue; }
        if (key == "tests") {
            for (const char* p = value.c_str(); *p;) {
                char* end;
                tests.push_back(std::strtoull(p, &end, 10));
                p = *end == ',' ? end + 1 : end;
            }
            continue;
        }
        if (!ApplyOverride(cfg, key, value)) { std::fprintf(stderr, "unknown setting %s\n", key.c_str()); return 1; }
        settings += " " + arg;
    }
    std::printf("seed %u, %llu steps, settings:%s\n", cfg.seed, steps, settings.empty() ? " (defaults)" : settings.c_str());

    World world(cfg);
    world.difficulty = startDifficulty;
    if (!seedCircuit.empty()) {
        // Start the population from hand-built brains instead of random ones.
        Config scratchCfg = cfg;
        scratchCfg.seed = cfg.seed + 12345;
        scratchCfg.minPopulation = 1;
        World scratch(scratchCfg);
        scratch.Step();
        HandParams hp;
        hp.circuit = seedCircuit;
        hp.wMotor = seedWMotor;
        std::mt19937 g(cfg.seed);
        std::uniform_real_distribution<float> ux(cfg.spawnMinX, cfg.worldWidth - 40.0f), uy(cfg.spawnMinY, cfg.spawnMaxY);
        for (int i = 0; i < seedCount; ++i) {
            Organism o = CloneOf(scratch.organisms[0], 500000 + uint64_t(i), cfg);
            o.brain = BuildCircuit(o, cfg, hp, cfg.seed * 1000 + uint32_t(i));
            o.brain.SetLearningRate(seedMix && i % 2 ? 0.0f : seedRate);
            o.mutationRate = 1.0f;  // CloneOf copies with mutation off; seeded lineages must be able to evolve
            o.x = ux(g);
            o.y = uy(g);
            o.health = cfg.maxHealth;
            o.air = cfg.maxAir;
            world.organisms.push_back(std::move(o));
        }
    }
    RowState st;
    uint64_t id = 10000000;
    size_t nextTest = 0;
    uint64_t spawnedAtWindowStart = 0;
    BiteTracker tracker;
    std::sort(tests.begin(), tests.end());
    for (unsigned long long i = 1; i <= steps; ++i) {
        const uint64_t p0 = world.stats.poisonBites, g0 = world.stats.goodBites;
        tracker.Before(world);
        world.Step();
        tracker.After(world, st);
        if (cfg.poison && cfg.poisonSwapInterval > 0) {
            const int bin = int(((world.stats.steps - 1) % uint64_t(cfg.poisonSwapInterval)) * 5 / uint64_t(cfg.poisonSwapInterval));
            st.phasePoison[bin] += world.stats.poisonBites - p0;
            st.phaseGood[bin] += world.stats.goodBites - g0;
        }
        if (foodCurriculum && i % foodWindow == 0) {
            const uint64_t refills = world.stats.spawned - spawnedAtWindowStart;
            spawnedAtWindowStart = world.stats.spawned;
            if (refills == 0) world.cfg.maxFood = std::max(foodMin, int(float(world.cfg.maxFood) * 0.9f));
            else if (refills >= 10) world.cfg.maxFood = std::min(cfg.maxFood, world.cfg.maxFood + 5);
        }
        if (i % report == 0) PrintRow(world, st);
        while (nextTest < tests.size() && tests[nextTest] == i) {
            const std::mt19937 saved = rng::Engine();  // tests don't change the evolution's random numbers
            std::vector<Organism> group;
            for (Organism& o : world.organisms) group.push_back(CloneOf(o, id++, world.cfg));
            char label[64];
            std::snprintf(label, sizeof(label), "step %llu", i);
            LearningTest(cfg, group, label);
            rng::Engine() = saved;
            ++nextTest;
        }
    }
    return 0;
}
