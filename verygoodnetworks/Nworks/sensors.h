#pragma once
#include "module.h"

struct Config;

// Things vision can tell apart. FoodB only exists with poison on; BigFood and
// Predator only with those features on.
enum class Seen { FoodA, FoodB, Air, Creature, BigFood, Predator, Count };

// Sector vision. The view is split into N slices: slice 0 starts at due left
// and they go clockwise (left, up, right, down for N = 4). For each kind of
// thing in use there are N inputs, each saying how close the nearest one in
// that slice is (1 = touching, 0 = nothing within visionRange). Air means air
// pockets and springs that still hold water.
class VisionSensor : public ModuleBase<VisionSensor> {
public:
    static constexpr int kMaxSectors = 32;

    VisionSensor(int sectors, const Config& cfg);
    int InputCount() const override { return sectors_ * kinds_; }
    void Sense(Organism& self, const World& world) override;

private:
    int sectors_;
    int kinds_ = 0;
    int slot_[int(Seen::Count)];  // input block for each kind, or -1 if unused
};

// Hearing: the loudest call (signal x closeness) from each of 4 directions:
// left, up, right, down.
class HearingSensor : public ModuleBase<HearingSensor> {
public:
    int InputCount() const override { return 4; }
    void Sense(Organism& self, const World& world) override;
};

// Taste: how good (input 0) or bad (input 1) the last thing eaten was,
// fading over a few steps. The same signal drives lifetime learning.
class TasteSensor : public ModuleBase<TasteSensor> {
public:
    int InputCount() const override { return 2; }
    void Sense(Organism& self, const World& world) override;
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

// Colour smell: how close the nearest food of each colour is (1 = touching,
// 0 = nothing within visionRange), whatever the direction. Unlike vision,
// one input carries "red is near" for every direction, so a single synapse
// can learn about a colour.
class ColourSmellSensor : public ModuleBase<ColourSmellSensor> {
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
