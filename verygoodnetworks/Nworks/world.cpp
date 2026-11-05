#include "world.h"

#include <algorithm>
#include <cmath>
#include "actuators.h"
#include "random.h"
#include "sensors.h"

namespace {

bool Touching(const Organism& o, const Resource& r) {
    const float dx = r.x - o.x;
    const float dy = r.y - o.y;
    const float reach = o.radius + r.radius;
    return dx * dx + dy * dy < reach * reach;
}

// Removes every item the organism touches and returns how many there were.
int Consume(std::vector<Resource>& items, const Organism& o) {
    const size_t before = items.size();
    items.erase(std::remove_if(items.begin(), items.end(), [&o](const Resource& r) { return Touching(o, r); }),
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
    for (int i = 0; i < cfg.initialFood; ++i) food.push_back(RandomResource(cfg.spawnMaxX, cfg.foodRadius));
    for (int i = 0; i < cfg.initialAirPockets; ++i)
        airPockets.push_back(RandomResource(cfg.spawnMaxX, cfg.airPocketRadius));
}

void World::Step() {
    ++stats.steps;

    for (Slice& s : slices) ++s.age;
    slices.erase(std::remove_if(slices.begin(), slices.end(), [](const Slice& s) { return s.IsExpired(); }),
                 slices.end());

    if (int(organisms.size()) < cfg.minPopulation) SpawnRandomOrganism();
    SpawnResources();

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

        if (o.health <= 0.0f || o.air <= 0.0f) o.alive = false;

        if (o.alive && o.foodEaten >= cfg.foodToReproduce) {
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
}

void World::SpawnRandomOrganism() {
    Organism o(nextId_++, rng::Float(cfg.spawnMinX, cfg.spawnMaxX), rng::Float(cfg.spawnMinY, cfg.spawnMaxY),
               cfg);
    o.AddModule(std::make_unique<FoodSensor>());
    o.AddModule(std::make_unique<AirPocketSensor>());
    o.AddModule(std::make_unique<VitalsSensor>());
    if (cfg.organismSensor) o.AddModule(std::make_unique<OrganismSensor>());
    if (cfg.positionSensor) o.AddModule(std::make_unique<PositionSensor>());
    if (cfg.slices) o.AddModule(std::make_unique<SliceSensor>());
    o.AddModule(std::make_unique<MovementModule>());
    if (cfg.slices) o.AddModule(std::make_unique<SliceModule>());

    o.brain = NeuralNetwork(o.InputsUsed(), o.OutputsUsed(), cfg.initialHiddenNeurons, cfg.initialConnections, cfg);
    organisms.push_back(std::move(o));
    ++stats.spawned;
}

void World::SpawnResources() {
    const float generation = float(MaxGeneration());

    if (int(food.size()) < cfg.maxFood && rng::Chance(cfg.foodSpawnChance)) {
        Resource r = RandomResource(cfg.spawnMaxX, cfg.foodRadius);
        r.x = std::max(0.0f, r.x - std::min(generation * cfg.foodDriftPerGeneration, cfg.foodDriftMax));
        food.push_back(r);
    }
    if (int(airPockets.size()) < cfg.maxAirPockets && rng::Chance(cfg.airSpawnChance)) {
        Resource r = RandomResource(cfg.airSpawnMaxX, cfg.airPocketRadius);
        r.x = std::min(cfg.worldWidth, r.x + std::min(generation * cfg.airDriftPerGeneration, cfg.airDriftMax));
        airPockets.push_back(r);
    }
}

Resource World::RandomResource(float maxX, float radius) const {
    return Resource{rng::Float(cfg.spawnMinX, maxX), rng::Float(cfg.spawnMinY, cfg.spawnMaxY), radius};
}

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
