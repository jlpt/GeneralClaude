#pragma once
#include "module.h"

// 4 outputs: right, left, down, up. Each step the organism moves by
// (right - left, down - up) * moveSpeed.
class MovementModule : public ModuleBase<MovementModule> {
public:
    int OutputCount() const override { return 4; }
    void Act(Organism& self, World& world) override;
};

// 1 output: a 0..1 call that nearby organisms can hear. Calling costs health.
class SignalModule : public ModuleBase<SignalModule> {
public:
    int OutputCount() const override { return 1; }
    void Act(Organism& self, World& world) override;
};

// 4 outputs: the two endpoints (x1, y1, x2, y2) of a cut, relative to the
// organism. When all four are active and the cooldown has passed, a slice
// is created; it kills any organism it touches.
class SliceModule : public ModuleBase<SliceModule> {
public:
    int OutputCount() const override { return 4; }
    void Act(Organism& self, World& world) override;

private:
    int cooldown_ = 0;
};
