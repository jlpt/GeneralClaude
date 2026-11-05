#pragma once
#include "module.h"

// Sector vision. The view is split into N slices: slice 0 starts at due left
// and they go clockwise (left, up, right, down for N = 4). Inputs are
// [food x N][air x N][organisms x N]; each is how close the nearest thing of
// that kind in that slice is (1 = touching, 0 = nothing within visionRange).
// Air means air pockets and springs that still hold water.
class VisionSensor : public ModuleBase<VisionSensor> {
public:
    static constexpr int kMaxSectors = 32;
    static constexpr int kKinds = 3;

    explicit VisionSensor(int sectors) : sectors_(sectors) {}
    int InputCount() const override { return sectors_ * kKinds; }
    void Sense(Organism& self, const World& world) override;

private:
    int sectors_;
};

// The original sensors (used when vision is off). Directional sensors (food, air pockets, other organisms) use 5 inputs:
//   left, right, up, down  - stronger the closer the target is on that axis
//   nothing                - 1 when nothing is in range
// They report the nearest target within range.

class FoodSensor : public ModuleBase<FoodSensor> {
public:
    int InputCount() const override { return 5; }
    void Sense(Organism& self, const World& world) override;
};

class AirPocketSensor : public ModuleBase<AirPocketSensor> {
public:
    int InputCount() const override { return 5; }
    void Sense(Organism& self, const World& world) override;
};

class OrganismSensor : public ModuleBase<OrganismSensor> {
public:
    int InputCount() const override { return 5; }
    void Sense(Organism& self, const World& world) override;
};

// Two urgency signals that reach 1 as the organism runs low:
//   hunger         - 1 at or below 30% health
//   breathlessness - 1 at or below 50% air
class VitalsSensor : public ModuleBase<VitalsSensor> {
public:
    int InputCount() const override { return 2; }
    void Sense(Organism& self, const World& world) override;
};

// Own position, scaled to 0..1 across the world.
class PositionSensor : public ModuleBase<PositionSensor> {
public:
    int InputCount() const override { return 2; }
    void Sense(Organism& self, const World& world) override;
};

// Both endpoints of the nearest slice: left, right, up, down for each (8 inputs).
class SliceSensor : public ModuleBase<SliceSensor> {
public:
    int InputCount() const override { return 8; }
    void Sense(Organism& self, const World& world) override;
};
