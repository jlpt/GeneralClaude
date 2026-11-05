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
    for (int i = 0; i < cfg.initialFood; ++i) SpawnFood();
    if (cfg.springs) {
        for (const auto& pos : kSpringLayout) {
            springs.push_back(
                Spring{pos[0] * cfg.worldWidth, pos[1] * cfg.worldHeight, cfg.springRadius, cfg.springCapacity});
        }
    } else {
        for (int i = 0; i < cfg.initialAirPockets; ++i)
            airPockets.push_back(RandomResource(cfg.spawnMaxX, cfg.airPocketRadius));
    }
}

void World::Step() {
    ++stats.steps;

    for (Slice& s : slices) ++s.age;
    slices.erase(std::remove_if(slices.begin(), slices.end(), [](const Slice& s) { return s.IsExpired(); }),
                 slices.end());

    if (int(organisms.size()) < cfg.minPopulation) RefillPopulation();
    if (int(food.size()) < cfg.maxFood && rng::Chance(cfg.foodSpawnChance)) SpawnFood();
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

        if (const int eaten = Consume(food, o); eaten > 0) {
            o.health = cfg.maxHealth;
            o.foodEaten += eaten;
        }
        if (Consume(airPockets, o) > 0) o.air = cfg.maxAir;
        for (Spring& s : springs) {
            if (!Touching(o, s.x, s.y, s.radius)) continue;
            const float drink = std::min(cfg.maxAir - o.air, s.water);
            o.air += drink;
            s.water -= drink;
        }

        if (o.health <= 0.0f || o.air <= 0.0f) o.alive = false;

        if (o.alive && o.foodEaten >= cfg.foodToReproduce) {
            RememberParent(o);
            newborns.push_back(o.MakeOffspring(nextId_++, cfg));
            o.foodEaten = 0;
            ++stats.births;
        }
    }

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
        o.brain.Mutate(int(float(o.brain.SynapseCount()) * cfg.mutationFraction * g.mutationRate), cfg);
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
        o.AddModule(std::make_unique<VisionSensor>(std::clamp(cfg.visionSectors, 1, VisionSensor::kMaxSectors)));
    } else {
        o.AddModule(std::make_unique<FoodSensor>());
        o.AddModule(std::make_unique<AirPocketSensor>());
    }
    o.AddModule(std::make_unique<VitalsSensor>());
    if (cfg.organismSensor) o.AddModule(std::make_unique<OrganismSensor>());
    if (cfg.positionSensor) o.AddModule(std::make_unique<PositionSensor>());
    if (cfg.slices) o.AddModule(std::make_unique<SliceSensor>());
    o.AddModule(std::make_unique<MovementModule>());
    if (cfg.slices) o.AddModule(std::make_unique<SliceModule>());
}

void World::SpawnFood() {
    Resource r{0.0f, 0.0f, cfg.foodRadius, stats.steps};
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
    food.push_back(r);
}

void World::SpawnAirPocket() {
    Resource r = RandomResource(cfg.airSpawnMaxX, cfg.airPocketRadius);
    r.x = std::min(cfg.worldWidth, r.x + difficulty * cfg.airDriftMax);
    airPockets.push_back(r);
}

Resource World::RandomResource(float maxX, float radius) const {
    return Resource{rng::Float(cfg.spawnMinX, maxX), rng::Float(cfg.spawnMinY, cfg.spawnMaxY), radius,
                    stats.steps};
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
