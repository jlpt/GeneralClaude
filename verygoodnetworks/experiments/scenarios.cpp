#include "scenarios.h"

#include <algorithm>
#include <cmath>
#include <random>
#include "actuators.h"

namespace {

constexpr float kTwoPi = 6.28318530718f;
constexpr float kCx = 650.0f, kCy = 384.0f;  // arena centre

bool IsFoodTest(Test t) {
    return t == Test::Forage || t == Test::ForageControl || t == Test::PoisonChoice || t == Test::MemoryNear ||
           t == Test::MemoryFar || t == Test::MemoryControl || t == Test::PoisonLearning;
}

// An empty arena: nothing spawns, no curriculum, poison does no damage (the
// bad taste still happens), no reproduction.
Config ArenaConfig(Test test, const Config& base, uint32_t seed) {
    Config c = base;
    c.seed = seed;
    c.minPopulation = 0;
    c.foodSpawnChance = 0.0f;
    c.bigFoodSpawnChance = 0.0f;
    c.airSpawnChance = 0.0f;
    c.curriculumWindow = 0;
    c.foodCurriculum = false;
    c.poisonDamage = 0.0f;
    c.foodToReproduce = 1000000000;
    c.maxAge = 0;
    const bool memory = test == Test::MemoryNear || test == Test::MemoryFar || test == Test::MemoryControl;
    c.foodVisibleSteps = memory ? 20 : 0;
    return c;
}

}  // namespace

const char* TestName(Test test) {
    switch (test) {
    case Test::Forage: return "forage";
    case Test::ForageControl: return "forage-control";
    case Test::PoisonChoice: return "poison-choice";
    case Test::Flee: return "flee";
    case Test::FleeControl: return "flee-control";
    case Test::MemoryNear: return "memory-150px";
    case Test::MemoryFar: return "memory-250px";
    case Test::MemoryControl: return "memory-control";
    case Test::Air: return "air";
    case Test::AirControl: return "air-control";
    case Test::PoisonLearning: return "poison-learning";
    default: return "?";
    }
}

Scenario::Scenario(Test test, const Config& base, uint32_t seed, Organism creature)
    : test_(test), world_(ArenaConfig(test, base, seed)), seed_(seed) {
    World& w = world_;
    w.food.clear();
    w.bigFood.clear();
    w.springs.clear();
    w.airPockets.clear();
    w.predators.clear();
    w.difficulty = (test == Test::Flee || test == Test::FleeControl) ? 0.5f : 0.0f;  // 1 predator at speed 1.9
    w.stats.steps = 1000;  // so "never visible" food can be older than foodVisibleSteps

    if (test == Test::PoisonLearning) w.poisonKindOverride = int(seed % 2);  // poisonous colour unknown in advance
    std::mt19937 g(seed * 2654435761u + 97u);
    const float a = std::uniform_real_distribution<float>(0.0f, kTwoPi)(g);
    const float dist = test == Test::PoisonChoice ? 120.0f
                       : test == Test::MemoryFar  ? 250.0f
                       : (test == Test::Air || test == Test::AirControl) ? 200.0f
                       : (test == Test::Flee || test == Test::FleeControl) ? 100.0f
                                                                           : 150.0f;
    tx_ = kCx + dist * std::cos(a);
    ty_ = kCy + dist * std::sin(a);

    const int bad = w.PoisonKind();
    const uint8_t good = uint8_t(w.cfg.poison ? 1 - bad : 0);
    const float r = w.cfg.foodRadius;
    switch (test) {
    case Test::Forage:
    case Test::MemoryNear:
    case Test::MemoryFar:
        w.food.push_back(Resource{tx_, ty_, r, w.stats.steps, good});
        break;
    case Test::MemoryControl:
        w.food.push_back(Resource{tx_, ty_, r, 0, good});  // born long ago: never visible
        break;
    case Test::PoisonChoice:
        w.food.push_back(Resource{tx_, ty_, r, w.stats.steps, good});
        w.food.push_back(Resource{2.0f * kCx - tx_, 2.0f * kCy - ty_, r, w.stats.steps, uint8_t(bad)});
        break;
    case Test::Flee:
        w.predators.push_back(Predator{tx_, ty_, a + 3.14159265f, 0});
        break;
    case Test::Air:
        w.springs.push_back(Spring{tx_, ty_, w.cfg.springRadius, w.cfg.springCapacity});
        break;
    case Test::PoisonLearning:
        PlaceChoicePair();
        break;
    default:
        break;
    }

    creature.x = kCx;
    creature.y = kCy;
    creature.health = IsFoodTest(test) ? 0.25f * w.cfg.maxHealth : w.cfg.maxHealth;
    creature.air = (test == Test::Air || test == Test::AirControl) ? 0.2f * w.cfg.maxAir : w.cfg.maxAir;
    creature.reward = 0.0f;
    creature.foodEaten = 0;
    w.organisms.push_back(std::move(creature));
    lastX_ = kCx;
    lastY_ = kCy;
}

// PoisonLearning: one food of each colour, 120 px from the arena centre on
// opposite sides, at an angle that changes every round.
void Scenario::PlaceChoicePair() {
    World& w = world_;
    std::mt19937 g(seed_ * 2654435761u + 31u * uint32_t(round_) + 7u);
    const float a = std::uniform_real_distribution<float>(0.0f, kTwoPi)(g);
    const int bad = w.PoisonKind();
    w.food.clear();
    w.food.push_back(Resource{kCx + 120.0f * std::cos(a), kCy + 120.0f * std::sin(a), w.cfg.foodRadius, w.stats.steps,
                              uint8_t(1 - bad)});
    w.food.push_back(Resource{kCx - 120.0f * std::cos(a), kCy - 120.0f * std::sin(a), w.cfg.foodRadius, w.stats.steps,
                              uint8_t(bad)});
}

void Scenario::Sense() {
    Organism* o = Creature();
    if (!o) return;
    for (auto& m : o->modules) m->Sense(*o, world_);
}

void Scenario::Finish(bool success) {
    done_ = true;
    success_ = success;
    if (test_ != Test::Flee && test_ != Test::FleeControl) score_ = success ? 1.0f : 0.0f;
}

void Scenario::Step() {
    if (done_) return;
    World& w = world_;
    const uint64_t good0 = w.stats.goodBites, poison0 = w.stats.poisonBites;
    w.Step();
    ++t_;
    Organism* o = Creature();
    if (o) {
        lastX_ = o->x;
        lastY_ = o->y;
    }
    const bool ateGood = w.stats.goodBites > good0, atePoison = w.stats.poisonBites > poison0;
    auto near = [&](float radius) {
        return o && std::hypot(o->x - tx_, o->y - ty_) < w.cfg.organismRadius + radius;
    };
    auto heading = [&] {  // cosine between movement so far and "away from the target point"
        const float dx = lastX_ - kCx, dy = lastY_ - kCy, len = std::hypot(dx, dy);
        const float ax = kCx - tx_, ay = kCy - ty_, alen = std::hypot(ax, ay);
        return len > 1.0f ? (dx * ax + dy * ay) / (len * alen) : 0.0f;
    };

    switch (test_) {
    case Test::Forage:
    case Test::MemoryNear:
    case Test::MemoryFar:
    case Test::MemoryControl:
        if (ateGood) return Finish(true);
        break;
    case Test::ForageControl:
        if (near(w.cfg.foodRadius)) return Finish(true);
        break;
    case Test::PoisonChoice:
        if (ateGood || atePoison) {
            bit_ = true;
            return Finish(ateGood && !atePoison);
        }
        break;
    case Test::Flee:
        if (t_ == 30 || (!o && t_ < 30)) score_ = heading();
        if (!o) return Finish(false);
        break;
    case Test::FleeControl:
        if (t_ == 30) {
            score_ = heading();
            return Finish(true);
        }
        break;
    case Test::Air:
    case Test::AirControl:
        if (near(w.cfg.springRadius)) return Finish(true);
        break;
    case Test::PoisonLearning: {
        if (!o) return Finish(false);
        if (pause_ > 0) {  // between rounds: let the taste sink in, then reset the creature and the food
            if (--pause_ == 0) {
                if (++round_ >= kLearningRounds) {
                    score_ = laterBites_ ? float(laterSafe_) / float(laterBites_) : 0.0f;
                    done_ = true;
                    success_ = laterBites_ > 0 && laterSafe_ * 2 > laterBites_;
                    return;
                }
                o->x = kCx;
                o->y = kCy;
                o->health = 0.25f * w.cfg.maxHealth;
                PlaceChoicePair();
                roundStart_ = t_;
            }
            return;
        }
        if (ateGood || atePoison) {
            if (round_ == 0) {  // first pick, before any taste of this world
                firstAte_ = bit_ = true;
                firstSafe_ = ateGood && !atePoison;
            } else {
                ++laterBites_;
                laterSafe_ += (ateGood && !atePoison) ? 1 : 0;
            }
            w.food.clear();
            pause_ = 30;
        } else if (t_ - roundStart_ >= 400) {
            w.food.clear();
            pause_ = 1;  // no bite this round: move on
        }
        return;  // its own time limit: 6 rounds of at most 430 steps
    }
    default:
        break;
    }
    if (!o) return Finish(false);
    if (t_ >= kTimeLimit) Finish(test_ == Test::Flee);  // surviving the whole trial counts for Flee
}

void ApplyAction(Organism& o, int action) {
    const int dir = action % 9;
    const bool eat = action / 9 == 1;
    float right = 0.0f, left = 0.0f, down = 0.0f, up = 0.0f;
    if (dir > 0) {
        const float theta = float(dir - 1) * kTwoPi / 8.0f;
        const float c = std::cos(theta), s = std::sin(theta);
        right = std::max(0.0f, c);
        left = std::max(0.0f, -c);
        down = std::max(0.0f, s);
        up = std::max(0.0f, -s);
    }
    for (const auto& m : o.modules) {
        if (dynamic_cast<const MovementModule*>(m.get())) {
            o.brain.SetOutput(m->firstOutput + 0, right);
            o.brain.SetOutput(m->firstOutput + 1, left);
            o.brain.SetOutput(m->firstOutput + 2, down);
            o.brain.SetOutput(m->firstOutput + 3, up);
        } else if (dynamic_cast<const EatModule*>(m.get())) {
            o.brain.SetOutput(m->firstOutput, eat ? 1.0f : 0.0f);
        } else if (dynamic_cast<const SignalModule*>(m.get())) {
            o.brain.SetOutput(m->firstOutput, 0.0f);
        }
    }
}
