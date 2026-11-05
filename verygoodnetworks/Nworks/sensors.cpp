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

void FoodSensor::Sense(Organism& self, const World& world) {
    SenseNearestResource(self, firstInput, world.food, world.cfg.resourceSensorRange);
}

void AirPocketSensor::Sense(Organism& self, const World& world) {
    SenseNearestResource(self, firstInput, world.airPockets, world.cfg.resourceSensorRange);
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
