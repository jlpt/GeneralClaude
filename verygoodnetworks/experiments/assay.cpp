// Behaviour tests for evolved Nworks creatures.
//
// 1. Evolves a world exactly like `nworks_headless --seed S --steps N` (all features on).
// 2. Copies every creature alive at the end (no mutation), plus the same number
//    of brand-new random brains.
// 3. Puts each copy alone in an empty arena and measures:
//      forage   - one food 150 px away (in view): eaten within 600 steps?
//                 control: same setup with no food; does it pass that spot anyway?
//      escape   - a predator 100 px away (1 predator, 50% difficulty speed):
//                 alive after 600 steps? and does it move away in the first 60 steps?
//      air      - 20% air, a spring 200 px away: reached within 600 steps?
//                 control: no spring; does it pass that spot anyway?
//      choice   - one food of each colour, 120 px away on opposite sides:
//                 which colour is eaten first?
// 4. Keeps running the evolved world and checks whether calls go with
//    predators or food (alarm/food calls).
//
// build: the nworks_assay target in CMakeLists.txt
// usage: assay SEED EVOLVE_STEPS [TRIALS_PER_BRAIN]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>
#include "organism.h"
#include "random.h"
#include "world.h"

namespace {

constexpr float kTwoPi = 6.28318530718f;
constexpr float kCx = 650.0f, kCy = 384.0f;  // arena centre

Config BaseConfig(uint32_t seed) {
    Config cfg;
    cfg.seed = seed;
    return cfg;
}

// An exact copy of a creature: MakeOffspring with mutation strength 0.
Organism CloneOf(Organism& src, uint64_t id, const Config& cfg) {
    const float saved = src.mutationRate;
    src.mutationRate = 0.0f;
    Organism c = src.MakeOffspring(id, cfg);
    src.mutationRate = saved;
    c.generation = src.generation;
    return c;
}

// An empty arena: nothing spawns, difficulty fixed, no population refill.
World MakeArena(const Config& base, uint32_t seed, float difficulty, bool poison) {
    Config c = base;
    c.seed = seed;
    c.minPopulation = 0;
    c.foodSpawnChance = 0.0f;
    c.bigFoodSpawnChance = 0.0f;
    c.airSpawnChance = 0.0f;
    c.curriculumWindow = 0;
    c.poison = poison;
    c.poisonSwapInterval = 1000000000;  // colour 0 stays poisonous for the whole test
    World w(c);
    w.food.clear();
    w.bigFood.clear();
    w.springs.clear();
    w.airPockets.clear();
    w.predators.clear();
    w.difficulty = difficulty;
    return w;
}

Organism& AddCreature(World& w, Organism& src, float x, float y) {
    Organism o = CloneOf(src, 1, w.cfg);
    o.x = x;
    o.y = y;
    o.health = w.cfg.maxHealth;
    o.air = w.cfg.maxAir;
    o.reward = 0.0f;
    o.foodEaten = 0;
    w.organisms.push_back(std::move(o));
    return w.organisms.back();
}

float Dist(float ax, float ay, float bx, float by) { return std::hypot(ax - bx, ay - by); }

struct Tally {
    int n = 0, hits = 0;
    double sum = 0.0;  // for averages
    void Add(bool hit, double value = 0.0) { ++n; hits += hit ? 1 : 0; sum += value; }
    double Rate() const { return n ? 100.0 * hits / n : 0.0; }
    double Mean() const { return n ? sum / n : 0.0; }
};

struct Results {
    Tally fleeControl, fleeCos, fleeCosControl, foodCos, foodCosControl;
    Tally forage, foragePoison, forageControl, escape, fleeing, towardFood, air, airControl, choiceSafe, choiceAny, speed;
};

// Trial geometry comes from its own generator so evolved and random brains
// face exactly the same layouts.
float TrialAngle(uint32_t seed) {
    std::mt19937 g(seed * 2654435761u + 7u);
    return std::uniform_real_distribution<float>(0.0f, kTwoPi)(g);
}

// Food of one colour 150 px away (in view). Also measures, after 60 steps,
// how much further the creature is from the food than at the start
// (negative = it moved toward the food).
void ForageOne(const Config& base, Organism& src, uint32_t seed, int kind, bool withFood, Tally& hits, Tally* gap,
               Tally* speed, Tally* towardCos = nullptr) {
    const float a = TrialAngle(seed);
    const float fx = kCx + 150.0f * std::cos(a), fy = kCy + 150.0f * std::sin(a);
    World w = MakeArena(base, seed, 0.0f, false);
    AddCreature(w, src, kCx, kCy).health = 0.25f * w.cfg.maxHealth;  // hungry
    if (withFood) w.food.push_back(Resource{fx, fy, w.cfg.foodRadius, 0, uint8_t(kind)});
    bool hit = false;
    float path = 0.0f, gapAt60 = 0.0f;
    int moved = 0;
    bool cosRecorded = false;
    for (int t = 0; t < 600 && !hit && !w.organisms.empty(); ++t) {
        const float px = w.organisms[0].x, py = w.organisms[0].y;
        w.Step();
        if (w.organisms.empty()) break;
        path += Dist(px, py, w.organisms[0].x, w.organisms[0].y);
        ++moved;
        if (withFood) hit = w.stats.goodBites > 0;
        else hit = Dist(w.organisms[0].x, w.organisms[0].y, fx, fy) < w.cfg.organismRadius + w.cfg.foodRadius;
        if (t == 59) gapAt60 = Dist(w.organisms[0].x, w.organisms[0].y, fx, fy) - 150.0f;
        if ((t == 29 || hit) && towardCos && !cosRecorded) {
            const float dx = w.organisms[0].x - kCx, dy = w.organisms[0].y - kCy, len = std::hypot(dx, dy);
            towardCos->Add(false, len > 1.0f ? (dx * std::cos(a) + dy * std::sin(a)) / len : 0.0f);
            cosRecorded = true;
        }
    }
    if (hit && moved < 60) gapAt60 = -150.0f;  // reached it before step 60
    hits.Add(hit);
    if (gap) gap->Add(false, gapAt60);
    if (speed && moved > 0) speed->Add(false, path / float(moved));
}

void Forage(const Config& base, Organism& src, uint32_t seed, int safeKind, Results& r) {
    ForageOne(base, src, seed, safeKind, true, r.forage, &r.towardFood, nullptr, &r.foodCos);
    ForageOne(base, src, seed, 1 - safeKind, true, r.foragePoison, nullptr, nullptr);
    ForageOne(base, src, seed, safeKind, false, r.forageControl, nullptr, &r.speed, &r.foodCosControl);
}

void EscapeOne(const Config& base, Organism& src, uint32_t seed, bool withPredator, Results& r) {
    const float a = TrialAngle(seed + 1000);
    const float px = kCx + 100.0f * std::cos(a), py = kCy + 100.0f * std::sin(a);
    World w = MakeArena(base, seed, 0.5f, false);  // 1 active predator at speed 1.9
    AddCreature(w, src, kCx, kCy);
    if (withPredator) w.predators.push_back(Predator{px, py, a + 3.14159f, 0});
    float gapAt60 = 100.0f, lastX = kCx, lastY = kCy, cosAway = 0.0f;
    bool cosDone = false;
    auto direction = [&](float x, float y) {  // cosine between movement so far and "away from the predator's start"
        const float dx = x - kCx, dy = y - kCy, len = std::hypot(dx, dy);
        return len > 1.0f ? (dx * -std::cos(a) + dy * -std::sin(a)) / len : 0.0f;
    };
    for (int t = 0; t < 600 && w.stats.eatenByPredators == 0 && !w.organisms.empty(); ++t) {
        w.Step();
        if (!w.organisms.empty()) { lastX = w.organisms[0].x; lastY = w.organisms[0].y; }
        if (t == 29) { cosAway = direction(lastX, lastY); cosDone = true; }
        // How far the creature moved from where the predator started (the predator's own movement doesn't count).
        if (t == 59 && !w.organisms.empty()) gapAt60 = Dist(w.organisms[0].x, w.organisms[0].y, px, py);
        if (!withPredator && t >= 59) break;
    }
    if (!cosDone) cosAway = direction(lastX, lastY);  // caught before step 30: use movement up to the catch
    (withPredator ? r.fleeCos : r.fleeCosControl).Add(false, cosAway);
    if (withPredator) {
        r.escape.Add(w.stats.eatenByPredators == 0);
        r.fleeing.Add(gapAt60 > 120.0f, gapAt60 - 100.0f);
    } else {
        r.fleeControl.Add(gapAt60 > 120.0f, gapAt60 - 100.0f);
    }
}

void Escape(const Config& base, Organism& src, uint32_t seed, Results& r) {
    EscapeOne(base, src, seed, true, r);
    EscapeOne(base, src, seed, false, r);
}

void Air(const Config& base, Organism& src, uint32_t seed, Results& r) {
    const float a = TrialAngle(seed + 2000);
    const float sx = kCx + 200.0f * std::cos(a), sy = kCy + 200.0f * std::sin(a);
    for (int withSpring = 0; withSpring < 2; ++withSpring) {
        World w = MakeArena(base, seed, 0.0f, false);
        AddCreature(w, src, kCx, kCy).air = 0.2f * w.cfg.maxAir;  // breathless
        if (withSpring) w.springs.push_back(Spring{sx, sy, w.cfg.springRadius, w.cfg.springCapacity});
        bool hit = false;
        for (int t = 0; t < 600 && !hit && !w.organisms.empty(); ++t) {
            w.Step();
            hit = !w.organisms.empty() &&
                  Dist(w.organisms[0].x, w.organisms[0].y, sx, sy) < w.cfg.organismRadius + w.cfg.springRadius;
        }
        (withSpring ? r.air : r.airControl).Add(hit);
    }
}

void Choice(const Config& base, Organism& src, uint32_t seed, int safeKindInEvolution, Results& r) {
    const float a = TrialAngle(seed + 3000);
    World w = MakeArena(base, seed, 0.0f, true);  // colour 0 is poisonous in the arena (no damage at 0% difficulty)
    AddCreature(w, src, kCx, kCy).health = 0.25f * w.cfg.maxHealth;
    // Colour that was safe at the end of evolution goes on one side, the other colour opposite.
    const int sideA = safeKindInEvolution, sideB = 1 - safeKindInEvolution;
    w.food.push_back(Resource{kCx + 120.0f * std::cos(a), kCy + 120.0f * std::sin(a), w.cfg.foodRadius, 0, uint8_t(sideA)});
    w.food.push_back(Resource{kCx - 120.0f * std::cos(a), kCy - 120.0f * std::sin(a), w.cfg.foodRadius, 0, uint8_t(sideB)});
    for (int t = 0; t < 600 && w.food.size() == 2 && !w.organisms.empty(); ++t) w.Step();
    if (w.food.size() < 2) {
        const int eatenKind = w.food.empty() ? sideA : 1 - w.food[0].kind;
        r.choiceAny.Add(true);
        r.choiceSafe.Add(eatenKind == safeKindInEvolution);
    } else {
        r.choiceAny.Add(false);
    }
}

void Report(const char* label, const Results& r) {
    std::printf("  %-14s speed %.2f px/step\n"
                "      forage (eats food 150 px away within 600 steps): recently-safe colour %5.1f%%, recently-poisonous colour %5.1f%%,"
                " no-food control %5.1f%% | heading toward safe food (cosine, first 30 steps) %+.2f, no-food control %+.2f\n"
                "      predator: survives 600 steps %5.1f%% | heading away from it (cosine, first 30 steps) %+.2f,"
                " same spot with no predator %+.2f\n"
                "      air (20%% air, spring 200 px away): reached %5.1f%% (no-spring control %5.1f%%)\n"
                "      choice (one of each colour): ate one %5.1f%%, of those picked the recently-safe colour %5.1f%%\n",
                label, r.speed.Mean(), r.forage.Rate(), r.foragePoison.Rate(), r.forageControl.Rate(), r.foodCos.Mean(), r.foodCosControl.Mean(),
                r.escape.Rate(), r.fleeCos.Mean(), r.fleeCosControl.Mean(), r.air.Rate(), r.airControl.Rate(), r.choiceAny.Rate(),
                r.choiceSafe.Rate());
    std::fflush(stdout);
}

Results TestGroup(const Config& base, std::vector<Organism>& group, int trials, int safeKind) {
    Results r;
    uint32_t seed = 1;
    for (Organism& src : group) {
        for (int t = 0; t < trials; ++t, ++seed) {
            Forage(base, src, seed, safeKind, r);
            Escape(base, src, seed, r);
            Air(base, src, seed, r);
            Choice(base, src, seed, safeKind, r);
        }
    }
    return r;
}

// Do calls go with predators or food? Samples every creature every step.
void CallStudy(World& w, int steps) {
    double callNearPredator = 0, nNearPredator = 0, callFoodClose = 0, nFoodClose = 0, callOther = 0, nOther = 0;
    double loud = 0, samples = 0;
    for (int s = 0; s < steps; ++s) {
        w.Step();
        for (const Organism& o : w.organisms) {
            bool predatorNear = false;
            for (int i = 0; i < w.ActivePredators(); ++i)
                if (Dist(o.x, o.y, w.predators[size_t(i)].x, w.predators[size_t(i)].y) < 200.0f) predatorNear = true;
            bool foodClose = false;
            for (const Resource& f : w.food)
                if (Dist(o.x, o.y, f.x, f.y) < 60.0f) { foodClose = true; break; }
            if (predatorNear) { callNearPredator += o.signal; ++nNearPredator; }
            else if (foodClose) { callFoodClose += o.signal; ++nFoodClose; }
            else { callOther += o.signal; ++nOther; }
            loud += o.signal > 0.2f ? 1 : 0;
            ++samples;
        }
    }
    auto avg = [](double s, double n) { return n > 0 ? s / n : 0.0; };
    std::printf("  calls: loud (>0.2) %.1f%% of the time | mean call with predator within 200 px %.3f (%0.f samples),"
                " food within 60 px %.3f, otherwise %.3f\n",
                100.0 * avg(loud, samples), avg(callNearPredator, nNearPredator), nNearPredator, avg(callFoodClose, nFoodClose),
                avg(callOther, nOther));
}

}  // namespace

void PrintRow(const World& w, uint64_t& lastPoison, uint64_t& lastGood) {
    double learn = 0.0;
    for (const Organism& o : w.organisms) learn += o.brain.LearningRate();
    const size_t n = w.organisms.size();
    const uint64_t p = w.stats.poisonBites - lastPoison, g = w.stats.goodBites - lastGood;
    lastPoison = w.stats.poisonBites;
    lastGood = w.stats.goodBites;
    std::printf("  step %9llu | alive %3zu | max gen %5d | diff %3.0f%% | neurons %6.1f | synapses %6.1f | poison bites %5.1f%% |"
                " learn %.4f | shared %6llu | eaten %6llu\n",
                (unsigned long long)w.stats.steps, n, w.MaxGeneration(), w.difficulty * 100.0f,
                n ? double(w.TotalNeurons()) / double(n) : 0.0, n ? double(w.TotalSynapses()) / double(n) : 0.0,
                p + g ? 100.0 * double(p) / double(p + g) : 0.0, n ? learn / double(n) : 0.0,
                (unsigned long long)w.stats.bigFoodShared, (unsigned long long)w.stats.eatenByPredators);
    std::fflush(stdout);
}

// Follows one population's colour preference across poison swaps:
// assay SEED track START END EVERY
int Track(uint32_t seed, unsigned long long start, unsigned long long end, unsigned long long every) {
    Config cfg = BaseConfig(seed);
    World world(cfg);
    uint64_t id = 1000000, lastPoison = 0, lastGood = 0;
    for (unsigned long long i = 0; i < start; ++i) world.Step();
    std::printf("seed %u colour tracking (hungry creature, one food 150 px away, eaten within 600 steps)\n", seed);
    for (unsigned long long at = start; at <= end; at += every) {
        while (world.stats.steps < at) world.Step();
        const int poisonNow = int(((world.stats.steps - 1) / uint64_t(cfg.poisonSwapInterval)) % 2);
        const std::mt19937 saved = rng::Engine();
        std::vector<Organism> evolved;
        for (Organism& o : world.organisms) evolved.push_back(CloneOf(o, id++, world.cfg));
        Tally c0, c1;
        uint32_t trialSeed = 1;
        for (Organism& src : evolved)
            for (int t = 0; t < 5; ++t, ++trialSeed) {
                ForageOne(cfg, src, trialSeed, 0, true, c0, nullptr, nullptr);
                ForageOne(cfg, src, trialSeed, 1, true, c1, nullptr, nullptr);
            }
        rng::Engine() = saved;
        const uint64_t p = world.stats.poisonBites - lastPoison, g = world.stats.goodBites - lastGood;
        lastPoison = world.stats.poisonBites;
        lastGood = world.stats.goodBites;
        std::printf("  step %9llu | poisonous colour now %d (for %5llu steps) | eats colour 0: %5.1f%%  colour 1: %5.1f%% |"
                    " world bites since last row poisonous %5.1f%% | alive %zu, avg learning rate %.4f\n",
                    (unsigned long long)world.stats.steps, poisonNow,
                    (unsigned long long)((world.stats.steps - 1) % uint64_t(cfg.poisonSwapInterval) + 1), c0.Rate(),
                    c1.Rate(), p + g ? 100.0 * double(p) / double(p + g) : 0.0, world.organisms.size(),
                    [&] { double s2 = 0; for (auto& o : world.organisms) s2 += o.brain.LearningRate();
                          return world.organisms.empty() ? 0.0 : s2 / double(world.organisms.size()); }());
        std::fflush(stdout);
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc >= 6 && std::string(argv[2]) == "track") {
        return Track(uint32_t(std::strtoul(argv[1], nullptr, 10)), std::strtoull(argv[3], nullptr, 10),
                     std::strtoull(argv[4], nullptr, 10), std::strtoull(argv[5], nullptr, 10));
    }
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s SEED CHECKPOINT[,CHECKPOINT...] [TRIALS_PER_BRAIN]\n", argv[0]);
        return 1;
    }
    const uint32_t seed = uint32_t(std::strtoul(argv[1], nullptr, 10));
    std::vector<unsigned long long> checkpoints;
    for (char* p = argv[2]; *p;) {
        checkpoints.push_back(std::strtoull(p, &p, 10));
        if (*p == ',') ++p;
    }
    const int trials = argc > 3 ? std::atoi(argv[3]) : 10;
    Config cfg = BaseConfig(seed);
    uint64_t id = 1000000;

    World world(cfg);
    const std::mt19937 afterWorldSeed = rng::Engine();

    // Baseline: brand-new random brains with the same body (40 of them).
    {
        Config rc = BaseConfig(seed + 777);
        rc.refillFromSurvivors = false;
        rc.minPopulation = 40;
        World scratch(rc);
        while (scratch.organisms.size() < 40) scratch.Step();
        std::vector<Organism> randomBrains;
        for (Organism& o : scratch.organisms) randomBrains.push_back(CloneOf(o, id++, rc));
        std::printf("seed %u baseline (40 random brains):\n", seed);
        Report("random brains", TestGroup(cfg, randomBrains, trials, 1));
    }
    rng::Engine() = afterWorldSeed;  // evolution continues exactly as nworks_headless --seed would

    uint64_t lastPoison = 0, lastGood = 0;
    unsigned long long done = 0;
    for (unsigned long long checkpoint : checkpoints) {
        for (; done < checkpoint; ++done) {
            world.Step();
            if ((done + 1) % 1000000 == 0) PrintRow(world, lastPoison, lastGood);
        }
        // Colour that was safe during the steps just before now (checkpoints can land on a swap).
        const int poisonRecently = int(((world.stats.steps - 1) / uint64_t(world.cfg.poisonSwapInterval)) % 2);
        const int safeKind = 1 - poisonRecently;
        std::printf("seed %u checkpoint %llu: %zu alive, max generation %d, difficulty %.0f%%, safe colour for the last"
                    " %llu steps: %d\n", seed, checkpoint, world.organisms.size(), world.MaxGeneration(),
                    world.difficulty * 100.0f,
                    (unsigned long long)((world.stats.steps - 1) % uint64_t(world.cfg.poisonSwapInterval) + 1), safeKind);
        const std::mt19937 saved = rng::Engine();  // tests must not change the evolution's random numbers
        std::vector<Organism> evolved;
        for (Organism& o : world.organisms) evolved.push_back(CloneOf(o, id++, world.cfg));
        Report("evolved", TestGroup(cfg, evolved, trials, safeKind));
        rng::Engine() = saved;
        std::fflush(stdout);
    }
    CallStudy(world, 200000);
    return 0;
}
