#include "world.h"

#include <algorithm>
#include <cmath>
#include "actuators.h"
#include "random.h"
#include "sensors.h"

namespace {

constexpr float kTwoPi = 6.28318530718f;

// Fixed layout, as fractions of the world size. The springs are more than
// two vision ranges apart; each food patch is out of sight of every spring.
constexpr float kSpringLayout[3][2] = {{0.115f, 0.195f}, {0.885f, 0.195f}, {0.5f, 0.846f}};
constexpr float kPatchLayout[World::kPatchCount][2] = {{0.5f, 0.26f}, {0.19f, 0.78f}, {0.81f, 0.78f}};

bool Touching(const Organism& o, float x, float y, float radius) {
    const float dx = x - o.x;
    const float dy = y - o.y;
    const float reach = o.radius + radius;
    return dx * dx + dy * dy < reach * reach;
}

// Removes every item the organism touches and returns how many there were.
int Consume(std::vector<Resource>& items, const Organism& o) {
    const size_t before = items.size();
    items.erase(std::remove_if(items.begin(), items.end(),
                               [&o](const Resource& r) { return Touching(o, r.x, r.y, r.radius); }),
                items.end());
    return int(before - items.size());
}

}  // namespace

// --- Slice ------------------------------------------------------------------

float Slice::Opacity() const {
    if (age < kFadeIn) return float(age) / float(kFadeIn);
    if (age < kFadeIn + kActive) return 1.0f;
    if (age < kFadeIn + kActive + kFadeOut) return 1.0f - float(age - kFadeIn - kActive) / float(kFadeOut);
    return 0.0f;
}

float Slice::DistanceTo(float x, float y) const {
    const float vx = x2 - x1, vy = y2 - y1;
    const float lengthSq = vx * vx + vy * vy;
    float t = lengthSq > 0.0f ? ((x - x1) * vx + (y - y1) * vy) / lengthSq : 0.0f;
    t = std::clamp(t, 0.0f, 1.0f);
    return std::hypot(x - (x1 + t * vx), y - (y1 + t * vy));
}

// --- World ------------------------------------------------------------------

World::World(const Config& config) : cfg(config) {
    rng::Seed(cfg.seed);
    for (int i = 0; i < cfg.initialFood * (cfg.poison ? 2 : 1); ++i) SpawnFood();
    if (cfg.springs) {
        for (const auto& pos : kSpringLayout) {
            springs.push_back(
                Spring{pos[0] * cfg.worldWidth, pos[1] * cfg.worldHeight, cfg.springRadius, cfg.springCapacity});
        }
    } else {
        for (int i = 0; i < cfg.initialAirPockets; ++i)
            airPockets.push_back(RandomResource(cfg.spawnMaxX, cfg.airPocketRadius));
    }
    if (cfg.predators) {
        for (int i = 0; i < cfg.predatorCount; ++i) {
            predators.push_back(Predator{rng::Float(0.0f, cfg.worldWidth), rng::Float(0.0f, cfg.worldHeight),
                                         rng::Float(0.0f, kTwoPi)});
        }
    }
}

void World::Step() {
    ++stats.steps;

    for (Slice& s : slices) ++s.age;
    slices.erase(std::remove_if(slices.begin(), slices.end(), [](const Slice& s) { return s.IsExpired(); }),
                 slices.end());

    if (int(organisms.size()) < cfg.minPopulation) RefillPopulation();
    const int foodColours = cfg.poison ? 2 : 1;  // poison lookalikes come on top of normal food
    if (int(food.size()) < cfg.maxFood * foodColours && rng::Chance(cfg.foodSpawnChance * float(foodColours)))
        SpawnFood();
    if (cfg.cooperativeFood && int(bigFood.size()) < cfg.maxBigFood && rng::Chance(cfg.bigFoodSpawnChance))
        SpawnBigFood();
    if (!cfg.springs && int(airPockets.size()) < cfg.maxAirPockets && rng::Chance(cfg.airSpawnChance))
        SpawnAirPocket();
    if (cfg.seasons && cfg.foodLifetime > 0) {
        const uint64_t now = stats.steps;
        const uint64_t lifetime = uint64_t(cfg.foodLifetime);
        food.erase(std::remove_if(food.begin(), food.end(),
                                  [now, lifetime](const Resource& r) { return now - r.bornStep > lifetime; }),
                   food.end());
    }
    for (Spring& s : springs) s.water = std::min(cfg.springCapacity, s.water + cfg.springRecharge);

    std::vector<Organism> newborns;
    for (Organism& o : organisms) {
        o.Update(*this);

        for (const Slice& s : slices) {
            if (s.IsActive() && s.DistanceTo(o.x, o.y) < o.radius + Slice::kWidth * 0.5f) {
                o.alive = false;
                break;
            }
        }

        EatFood(o);
        if (Consume(airPockets, o) > 0) o.air = cfg.maxAir;
        for (Spring& s : springs) {
            if (!Touching(o, s.x, s.y, s.radius)) continue;
            const float drink = std::min(cfg.maxAir - o.air, s.water);
            o.air += drink;
            s.water -= drink;
        }

        if (o.health <= 0.0f || o.air <= 0.0f) o.alive = false;
        if (cfg.maxAge > 0 && o.age >= cfg.maxAge) o.alive = false;

        if (o.alive && o.foodEaten >= cfg.foodToReproduce) {
            RememberParent(o);
            newborns.push_back(o.MakeOffspring(nextId_++, cfg));
            o.foodEaten = 0;
            ++stats.births;
        }
    }
    ShareBigFood();
    MovePredators();

    const size_t before = organisms.size();
    organisms.erase(std::remove_if(organisms.begin(), organisms.end(), [](const Organism& o) { return !o.alive; }),
                    organisms.end());
    stats.deaths += before - organisms.size();
    for (Organism& child : newborns) organisms.push_back(std::move(child));

    for (const Organism& o : organisms) stats.largestBrainEver = std::max(stats.largestBrainEver, o.brain.NeuronCount());
    UpdateCurriculum();
}

void World::UpdateCurriculum() {
    if (cfg.curriculumWindow <= 0 || stats.steps % uint64_t(cfg.curriculumWindow) != 0) return;
    if (spawnedThisWindow_ == 0) {
        difficulty = std::min(1.0f, difficulty + cfg.curriculumStep);  // self-sustaining: harder
    } else if (spawnedThisWindow_ >= uint64_t(cfg.curriculumBackoffSpawns)) {
        difficulty = std::max(0.0f, difficulty - cfg.curriculumStep);  // collapsing: easier
    }
    spawnedThisWindow_ = 0;
}

void World::RefillPopulation() {
    Organism o(nextId_++, rng::Float(cfg.spawnMinX, cfg.spawnMaxX), rng::Float(cfg.spawnMinY, cfg.spawnMaxY),
               cfg);
    AddBodyModules(o);

    if (cfg.refillFromSurvivors && !archive_.empty() && !rng::Chance(cfg.randomRefillShare)) {
        // A mutated copy of a recent parent, like a seed that waited out the crash.
        const Genome& g = archive_[size_t(rng::Int(0, int(archive_.size()) - 1))];
        o.brain = g.brain;
        o.brain.Mutate(rng::RoundRandomly(float(o.brain.SynapseCount()) * cfg.mutationFraction * g.mutationRate), cfg);
        o.brain.ResetState();
        o.mutationRate = g.mutationRate;
        o.color = g.color;
        o.generation = g.generation + 1;
    } else {
        o.brain =
            NeuralNetwork(o.InputsUsed(), o.OutputsUsed(), cfg.initialHiddenNeurons, cfg.initialConnections, cfg);
    }
    organisms.push_back(std::move(o));
    ++stats.spawned;
    ++spawnedThisWindow_;
}

void World::RememberParent(const Organism& parent) {
    if (!cfg.refillFromSurvivors || cfg.survivorArchiveSize <= 0) return;
    Genome g{parent.brain, parent.mutationRate, parent.color, parent.generation};
    if (archive_.size() < size_t(cfg.survivorArchiveSize)) {
        archive_.push_back(std::move(g));
    } else {
        archive_[archiveNext_] = std::move(g);
        archiveNext_ = (archiveNext_ + 1) % archive_.size();
    }
}

void World::AddBodyModules(Organism& o) const {
    if (cfg.vision) {
        o.AddModule(
            std::make_unique<VisionSensor>(std::clamp(cfg.visionSectors, 1, VisionSensor::kMaxSectors), cfg));
    } else {
        o.AddModule(std::make_unique<FoodSensor>());
        o.AddModule(std::make_unique<AirPocketSensor>());
    }
    o.AddModule(std::make_unique<VitalsSensor>());
    if (cfg.poison || cfg.plasticity) o.AddModule(std::make_unique<TasteSensor>());
    if (cfg.signals) o.AddModule(std::make_unique<HearingSensor>());
    if (cfg.colourSmell) o.AddModule(std::make_unique<ColourSmellSensor>());
    if (cfg.organismSensor) o.AddModule(std::make_unique<OrganismSensor>());
    if (cfg.positionSensor) o.AddModule(std::make_unique<PositionSensor>());
    if (cfg.slices) o.AddModule(std::make_unique<SliceSensor>());
    o.AddModule(std::make_unique<MovementModule>());
    if (cfg.signals) o.AddModule(std::make_unique<SignalModule>());
    if (cfg.slices) o.AddModule(std::make_unique<SliceModule>());
}

void World::EatFood(Organism& o) {
    const int poisonKind = cfg.poison ? PoisonKind() : -1;
    for (auto it = food.begin(); it != food.end();) {
        if (!Touching(o, it->x, it->y, it->radius)) {
            ++it;
            continue;
        }
        if (it->kind == poisonKind) {
            o.health -= cfg.poisonDamage * (cfg.poisonScalesWithDifficulty ? difficulty : 1.0f) * cfg.maxHealth;
            o.reward = std::max(-1.0f, o.reward - 1.0f);
            ++o.poisonBitesInLife;
            ++stats.poisonBites;
        } else {
            o.health = cfg.maxHealth;
            ++o.foodEaten;
            o.reward = std::min(1.0f, o.reward + 1.0f);
            ++o.goodBitesInLife;
            ++stats.goodBites;
        }
        it = food.erase(it);
    }
}

void World::ShareBigFood() {
    std::vector<Organism*> touching;
    for (auto it = bigFood.begin(); it != bigFood.end();) {
        touching.clear();
        for (Organism& o : organisms) {
            if (o.alive && Touching(o, it->x, it->y, it->radius)) touching.push_back(&o);
        }
        if (int(touching.size()) < cfg.bigFoodPartners) {
            ++it;
            continue;
        }
        for (Organism* o : touching) {
            o->health = cfg.maxHealth;
            o->foodEaten += cfg.bigFoodValue;
            o->reward = 1.0f;
        }
        ++stats.bigFoodShared;
        it = bigFood.erase(it);
    }
}

void World::MovePredators() {
    const float speed = PredatorSpeed();
    const float sightSq = cfg.predatorSight * cfg.predatorSight;
    const int active = ActivePredators();
    for (int i = 0; i < active; ++i) {
        Predator& p = predators[size_t(i)];
        if (p.rest > 0) {
            --p.rest;
            continue;
        }
        const Organism* target = nullptr;
        float bestSq = sightSq;
        for (const Organism& o : organisms) {
            const float dx = o.x - p.x, dy = o.y - p.y;
            if (o.alive && dx * dx + dy * dy < bestSq) {
                bestSq = dx * dx + dy * dy;
                target = &o;
            }
        }
        if (target) {
            // Chase at full speed.
            const float dx = target->x - p.x, dy = target->y - p.y;
            const float length = std::max(1e-3f, std::sqrt(dx * dx + dy * dy));
            p.x += dx / length * speed;
            p.y += dy / length * speed;
            p.heading = std::atan2(dy, dx);
        } else {
            // Wander at half speed, bouncing off the walls.
            p.heading += rng::Float(-0.3f, 0.3f);
            p.x += std::cos(p.heading) * speed * 0.5f;
            p.y += std::sin(p.heading) * speed * 0.5f;
            if (p.x < 0.0f || p.x > cfg.worldWidth) p.heading = kTwoPi * 0.5f - p.heading;
            if (p.y < 0.0f || p.y > cfg.worldHeight) p.heading = -p.heading;
        }
        p.x = std::clamp(p.x, 0.0f, cfg.worldWidth);
        p.y = std::clamp(p.y, 0.0f, cfg.worldHeight);

        for (Organism& o : organisms) {
            if (o.alive && Touching(o, p.x, p.y, cfg.predatorRadius)) {
                o.alive = false;
                ++stats.eatenByPredators;
                p.rest = cfg.predatorRest;
                break;
            }
        }
    }
}

void World::SpawnBigFood() {
    const float margin = 30.0f;
    bigFood.push_back(Resource{rng::Float(margin, cfg.worldWidth - margin), rng::Float(margin, cfg.worldHeight - margin),
                               cfg.bigFoodRadius, stats.steps, 0});
}

int World::PoisonKind() const {
    if (cfg.poisonSwapInterval <= 0) return 0;
    return int((stats.steps / uint64_t(cfg.poisonSwapInterval)) % 2);
}

int World::ActivePredators() const {
    return std::min(int(predators.size()), int(std::lround(difficulty * float(cfg.predatorCount))));
}

float World::PredatorSpeed() const { return cfg.predatorBaseSpeed + difficulty * cfg.predatorExtraSpeed; }

void World::SpawnFood() {
    Resource r{0.0f, 0.0f, cfg.foodRadius, stats.steps, 0};
    if (cfg.seasons) {
        const float margin = 20.0f;
        if (rng::Chance(difficulty)) {
            // Inside the blooming patch (uniform over the disc).
            const int patch = Season();
            const float angle = rng::Float(0.0f, kTwoPi);
            const float distance = cfg.patchRadius * std::sqrt(rng::Float(0.0f, 1.0f));
            r.x = PatchX(patch) + std::cos(angle) * distance;
            r.y = PatchY(patch) + std::sin(angle) * distance;
        } else {
            r.x = rng::Float(margin, cfg.worldWidth - margin);
            r.y = rng::Float(margin, cfg.worldHeight - margin);
        }
        r.x = std::clamp(r.x, margin, cfg.worldWidth - margin);
        r.y = std::clamp(r.y, margin, cfg.worldHeight - margin);
    } else {
        r = RandomResource(cfg.spawnMaxX, cfg.foodRadius);
        r.x = std::max(0.0f, r.x - difficulty * cfg.foodDriftMax);
    }
    if (cfg.poison) r.kind = uint8_t(rng::Int(0, 1));
    food.push_back(r);
}

void World::SpawnAirPocket() {
    Resource r = RandomResource(cfg.airSpawnMaxX, cfg.airPocketRadius);
    r.x = std::min(cfg.worldWidth, r.x + difficulty * cfg.airDriftMax);
    airPockets.push_back(r);
}

Resource World::RandomResource(float maxX, float radius) const {
    return Resource{rng::Float(cfg.spawnMinX, maxX), rng::Float(cfg.spawnMinY, cfg.spawnMaxY), radius,
                    stats.steps, 0};
}

int World::Season() const {
    if (cfg.seasonLength <= 0) return 0;
    return int((stats.steps / uint64_t(cfg.seasonLength)) % uint64_t(kPatchCount));
}

float World::PatchX(int patch) const { return kPatchLayout[patch][0] * cfg.worldWidth; }

float World::PatchY(int patch) const { return kPatchLayout[patch][1] * cfg.worldHeight; }

const Organism* World::Find(uint64_t id) const {
    for (const Organism& o : organisms) {
        if (o.id == id) return &o;
    }
    return nullptr;
}

const Organism* World::Leader() const {
    const Organism* leader = nullptr;
    for (const Organism& o : organisms) {
        if (!leader || o.generation > leader->generation) leader = &o;
    }
    return leader;
}

const Organism* World::Largest() const {
    const Organism* largest = nullptr;
    for (const Organism& o : organisms) {
        if (!largest || o.brain.NeuronCount() > largest->brain.NeuronCount()) largest = &o;
    }
    return largest;
}

int World::MaxGeneration() const {
    const Organism* leader = Leader();
    return leader ? leader->generation : 0;
}

size_t World::TotalNeurons() const {
    size_t total = 0;
    for (const Organism& o : organisms) total += o.brain.NeuronCount();
    return total;
}

size_t World::TotalSynapses() const {
    size_t total = 0;
    for (const Organism& o : organisms) total += o.brain.SynapseCount();
    return total;
}
