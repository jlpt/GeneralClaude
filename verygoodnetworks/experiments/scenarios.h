// Intelligence test battery: short trials with one creature alone in an empty
// arena. The same scenarios are used for evolved brains (nworks_iq) and for
// brains trained outside Nworks (the Python bridge), so their scores compare.
#pragma once
#include <cstdint>
#include "organism.h"
#include "world.h"

enum class Test : int {
    Forage = 0,     // hungry, one safe (purple) food 150 px away, always visible: eaten within 600 steps?
    ForageControl,  // same, no food: does it touch that spot anyway?
    PoisonChoice,   // one purple and one red food, 120 px away on opposite sides: which is eaten first?
    Flee,           // a predator 100 px away: alive after 600 steps? heading away in the first 30 steps?
    FleeControl,    // same spot, no predator: heading relative to it (should be ~0)
    MemoryNear,     // purple food 150 px away, visible only for the first 20 steps: eaten within 600?
    MemoryFar,      // same, 250 px away
    MemoryControl,  // purple food 150 px away, never visible: found by chance?
    Air,            // 20% air, a spring 200 px away: reached?
    AirControl,     // same, no spring: does it touch that spot anyway?
    PoisonLearning, // which colour is poisonous is random; 6 rounds of one red + one purple (120 px apart):
                    // success = safe picks in rounds 2-6, after the first taste
    Count
};
const char* TestName(Test test);

class Scenario {
public:
    static constexpr int kTimeLimit = 600;
    static constexpr int kLearningRounds = 6;

    // `creature` is placed in the middle of a fresh arena built from `base`.
    Scenario(Test test, const Config& base, uint32_t seed, Organism creature);

    void Step();                       // one simulation step; nothing once done
    bool Done() const { return done_; }
    bool Success() const { return success_; }
    float Score() const { return score_; }  // heading cosine for Flee/FleeControl, else 1 or 0
    bool Bit() const { return bit_; }       // PoisonChoice: ate one of the two
    // PoisonLearning: the first round's pick, and bites / safe picks in rounds 2-6.
    bool FirstAte() const { return firstAte_; }
    bool FirstSafe() const { return firstSafe_; }
    int LaterBites() const { return laterBites_; }
    int LaterSafe() const { return laterSafe_; }
    Organism* Creature() { return world_.organisms.empty() ? nullptr : &world_.organisms[0]; }
    void Sense();                      // refresh the creature's inputs (external control)

private:
    void Finish(bool success);
    void PlaceChoicePair();

    Test test_;
    World world_;
    float tx_ = 0.0f, ty_ = 0.0f;      // target point (food, spring or predator start)
    float lastX_ = 0.0f, lastY_ = 0.0f;
    int t_ = 0;
    bool done_ = false, success_ = false, bit_ = false;
    float score_ = 0.0f;
    int round_ = 0, roundStart_ = 0, pause_ = 0, laterBites_ = 0, laterSafe_ = 0;
    bool firstAte_ = false, firstSafe_ = false;
    uint32_t seed_ = 0;
};

// Discrete actions for externally controlled creatures: 9 movement choices
// (stay, or full speed in one of 8 directions) x eat (no / yes).
constexpr int kNumActions = 18;
void ApplyAction(Organism& o, int action);
// Many-creature worlds add a call bit: actions 0..35 = ApplyAction(a % 18), calling if a >= 18.
constexpr int kNumActionsWithCall = 36;
void ApplyActionWithCall(Organism& o, int action);
