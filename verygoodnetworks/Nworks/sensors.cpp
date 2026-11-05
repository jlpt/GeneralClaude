#include "sensors.h"

#include <algorithm>
#include <cmath>
#include <vector>
#include "organism.h"
#include "world.h"

namespace {

// Writes the 5 directional inputs. dx/dy point from the organism to the target.
void WriteDirection(Organism& self, int first, bool found, float dx, float dy, float range) {
    float left = 0.0f, right = 0.0f, up = 0.0f, down = 0.0f;
    if (found) {
        const float strengthX = std::max(0.0f, (range - std::fabs(dx)) / range);
        const float strengthY = std::max(0.0f, (range - std::fabs(dy)) / range);
        (dx < 0.0f ? left : right) = strengthX;
        (dy < 0.0f ? up : down) = strengthY;
    }
    self.brain.SetInput(first + 0, left);
    self.brain.SetInput(first + 1, right);
    self.brain.SetInput(first + 2, up);
    self.brain.SetInput(first + 3, down);
    self.brain.SetInput(first + 4, found ? 0.0f : 1.0f);
}

void SenseNearestResource(Organism& self, int first, const std::vector<Resource>& items, float range) {
    const Resource* nearest = nullptr;
    float bestSq = range * range;
    for (const Resource& r : items) {
        const float dx = r.x - self.x;
        const float dy = r.y - self.y;
        const float distSq = dx * dx + dy * dy;
        if (distSq < bestSq) {
            bestSq = distSq;
            nearest = &r;
        }
    }
    if (nearest) WriteDirection(self, first, true, nearest->x - self.x, nearest->y - self.y, range);
    else WriteDirection(self, first, false, 0.0f, 0.0f, range);
}

// Left/right/up/down strengths for one point; can't go negative.
void WritePoint(Organism& self, int first, float dx, float dy, float range) {
    const float strengthX = std::max(0.0f, (range - std::fabs(dx)) / range);
    const float strengthY = std::max(0.0f, (range - std::fabs(dy)) / range);
    self.brain.SetInput(first + 0, dx < 0.0f ? strengthX : 0.0f);
    self.brain.SetInput(first + 1, dx < 0.0f ? 0.0f : strengthX);
    self.brain.SetInput(first + 2, dy < 0.0f ? strengthY : 0.0f);
    self.brain.SetInput(first + 3, dy < 0.0f ? 0.0f : strengthY);
}

}  // namespace

// Which of `sectors` equal slices (dx, dy) falls in. atan2 is -pi at due left
// and increases clockwise on screen, because y points down.
static int SectorOf(float dx, float dy, int sectors) {
    constexpr float kPi = 3.14159265359f;
    const int sector = int((std::atan2(dy, dx) + kPi) / (2.0f * kPi) * float(sectors));
    return std::clamp(sector, 0, sectors - 1);
}

VisionSensor::VisionSensor(int sectors, const Config& cfg) : sectors_(sectors) {
    const bool used[int(Seen::Count)] = {true, cfg.poison, true, true, cfg.cooperativeFood, cfg.predators};
    for (int kind = 0; kind < int(Seen::Count); ++kind) slot_[kind] = used[kind] ? kinds_++ : -1;
}

void VisionSensor::Sense(Organism& self, const World& world) {
    const float range = world.cfg.visionRange;
    const float rangeSq = range * range;
    float seen[int(Seen::Count)][kMaxSectors] = {};

    auto look = [&](Seen kind, float x, float y, float radius) {
        const int slot = slot_[int(kind)];
        const float dx = x - self.x;
        const float dy = y - self.y;
        const float distSq = dx * dx + dy * dy;
        if (slot < 0 || distSq >= rangeSq) return;
        const float closeness = 1.0f - std::max(0.0f, std::sqrt(distSq) - radius) / range;
        float& cell = seen[slot][SectorOf(dx, dy, sectors_)];
        cell = std::max(cell, closeness);
    };

    for (const Resource& r : world.food) look(r.kind == 1 ? Seen::FoodB : Seen::FoodA, r.x, r.y, r.radius);
    for (const Resource& r : world.airPockets) look(Seen::Air, r.x, r.y, r.radius);
    for (const Spring& s : world.springs) {
        if (s.water >= 1.0f) look(Seen::Air, s.x, s.y, s.radius);
    }
    for (const Organism& other : world.organisms) {
        if (other.id != self.id) look(Seen::Creature, other.x, other.y, other.radius);
    }
    for (const Resource& r : world.bigFood) look(Seen::BigFood, r.x, r.y, r.radius);
    for (int i = 0; i < world.ActivePredators(); ++i) {
        const Predator& p = world.predators[size_t(i)];
        look(Seen::Predator, p.x, p.y, world.cfg.predatorRadius);
    }

    for (int kind = 0; kind < kinds_; ++kind) {
        for (int s = 0; s < sectors_; ++s) self.brain.SetInput(firstInput + kind * sectors_ + s, seen[kind][s]);
    }
}

void HearingSensor::Sense(Organism& self, const World& world) {
    const float range = world.cfg.hearingRange;
    float loudest[4] = {};
    for (const Organism& other : world.organisms) {
        if (other.id == self.id || other.signal <= 0.0f) continue;
        const float dx = other.x - self.x;
        const float dy = other.y - self.y;
        const float distSq = dx * dx + dy * dy;
        if (distSq >= range * range) continue;
        float& cell = loudest[SectorOf(dx, dy, 4)];
        cell = std::max(cell, other.signal * (1.0f - std::sqrt(distSq) / range));
    }
    for (int i = 0; i < 4; ++i) self.brain.SetInput(firstInput + i, loudest[i]);
}

void TasteSensor::Sense(Organism& self, const World& /*world*/) {
    self.brain.SetInput(firstInput + 0, std::max(0.0f, self.reward));
    self.brain.SetInput(firstInput + 1, std::max(0.0f, -self.reward));
}

void FoodSensor::Sense(Organism& self, const World& world) {
    SenseNearestResource(self, firstInput, world.food, world.cfg.resourceSensorRange);
}

void AirPocketSensor::Sense(Organism& self, const World& world) {
    // Springs count as air too, so this sensor still works with springs on.
    const float range = world.cfg.resourceSensorRange;
    float bestSq = range * range;
    float bestDx = 0.0f, bestDy = 0.0f;
    bool found = false;
    auto consider = [&](float x, float y) {
        const float dx = x - self.x, dy = y - self.y;
        if (dx * dx + dy * dy < bestSq) {
            bestSq = dx * dx + dy * dy;
            bestDx = dx;
            bestDy = dy;
            found = true;
        }
    };
    for (const Resource& r : world.airPockets) consider(r.x, r.y);
    for (const Spring& s : world.springs) {
        if (s.water >= 1.0f) consider(s.x, s.y);
    }
    WriteDirection(self, firstInput, found, bestDx, bestDy, range);
}

void OrganismSensor::Sense(Organism& self, const World& world) {
    const float range = world.cfg.organismSensorRange;
    const Organism* nearest = nullptr;
    float bestSq = range * range;
    for (const Organism& other : world.organisms) {
        if (other.id == self.id) continue;
        const float dx = other.x - self.x;
        const float dy = other.y - self.y;
        const float distSq = dx * dx + dy * dy;
        if (distSq < bestSq) {
            bestSq = distSq;
            nearest = &other;
        }
    }
    if (nearest) WriteDirection(self, firstInput, true, nearest->x - self.x, nearest->y - self.y, range);
    else WriteDirection(self, firstInput, false, 0.0f, 0.0f, range);
}

void VitalsSensor::Sense(Organism& self, const World& world) {
    const float hungerLevel = 0.3f * world.cfg.maxHealth;
    const float breathLevel = 0.5f * world.cfg.maxAir;
    self.brain.SetInput(firstInput + 0, std::min(1.0f, hungerLevel / std::max(self.health, 1.0f)));
    self.brain.SetInput(firstInput + 1, std::min(1.0f, breathLevel / std::max(self.air, 1.0f)));
}

void PositionSensor::Sense(Organism& self, const World& world) {
    self.brain.SetInput(firstInput + 0, self.x / world.cfg.worldWidth);
    self.brain.SetInput(firstInput + 1, self.y / world.cfg.worldHeight);
}

void SliceSensor::Sense(Organism& self, const World& world) {
    const float range = world.cfg.sliceSensorRange;
    const Slice* nearest = nullptr;
    float best = range;
    for (const Slice& s : world.slices) {
        const float distance = s.DistanceTo(self.x, self.y);
        if (distance < best) {
            best = distance;
            nearest = &s;
        }
    }
    if (nearest) {
        WritePoint(self, firstInput + 0, nearest->x1 - self.x, nearest->y1 - self.y, range);
        WritePoint(self, firstInput + 4, nearest->x2 - self.x, nearest->y2 - self.y, range);
    } else {
        for (int i = 0; i < InputCount(); ++i) self.brain.SetInput(firstInput + i, 0.0f);
    }
}
