#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include "config.h"
#include "organism.h"

// Food (restores health) or an air pocket (restores air).
struct Resource {
    float x, y;
    float radius;
};

// A short-lived cut made by a SliceModule. It fades in, kills any organism it
// touches during its single active step, then fades out.
struct Slice {
    static constexpr float kWidth = 5.0f;
    static constexpr int kFadeIn = 10;
    static constexpr int kActive = 1;
    static constexpr int kFadeOut = 10;

    float x1, y1, x2, y2;
    int age = 0;

    bool IsActive() const { return age >= kFadeIn && age < kFadeIn + kActive; }
    bool IsExpired() const { return age > kFadeIn + kActive + kFadeOut; }
    float Opacity() const;                    // 0..1, for drawing
    float DistanceTo(float x, float y) const; // from a point to the segment
};

struct WorldStats {
    uint64_t steps = 0;
    uint64_t births = 0;   // offspring from reproduction
    uint64_t spawned = 0;  // random organisms added to keep the population up
    uint64_t deaths = 0;
};

class World {
public:
    explicit World(const Config& config);

    void Step();

    const Organism* Find(uint64_t id) const;
    const Organism* Leader() const;  // the organism with the highest generation
    int MaxGeneration() const;
    size_t TotalNeurons() const;
    size_t TotalSynapses() const;

    Config cfg;
    std::vector<Organism> organisms;
    std::vector<Resource> food;
    std::vector<Resource> airPockets;
    std::vector<Slice> slices;
    WorldStats stats;

private:
    void SpawnRandomOrganism();
    void SpawnResources();
    Resource RandomResource(float maxX, float radius) const;

    uint64_t nextId_ = 1;
};
