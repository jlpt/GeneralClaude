#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include "config.h"
#include "organism.h"

// Food (restores health), big food, or an air pocket (restores air).
struct Resource {
    float x, y;
    float radius;
    uint64_t bornStep = 0;  // for rotting food
    uint8_t kind = 0;       // food colour, 0 or 1 (with poison on, one of them is poisonous)
};

// Hunts organisms; not evolved.
struct Predator {
    float x, y;
    float heading;  // radians, used while wandering
    int rest = 0;   // steps left resting after a catch
};

// A fixed air source with limited, slowly refilling capacity.
struct Spring {
    float x, y;
    float radius;
    float water;  // air units available
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
    uint64_t spawned = 0;  // organisms added to keep the population up (from the seed bank or random)
    uint64_t deaths = 0;
    uint64_t eatenByPredators = 0;
    uint64_t poisonBites = 0;
    uint64_t goodBites = 0;
    uint64_t bigFoodShared = 0;   // big food items opened together
    size_t largestBrainEver = 0;  // neurons
};

class World {
public:
    static constexpr int kPatchCount = 3;

    explicit World(const Config& config);

    void Step();

    const Organism* Find(uint64_t id) const;
    const Organism* Leader() const;   // the organism with the highest generation
    const Organism* Largest() const;  // the organism with the most neurons
    int MaxGeneration() const;
    size_t TotalNeurons() const;
    size_t TotalSynapses() const;

    // Seasons: which patch is blooming, and where the patches are.
    int Season() const;
    float PatchX(int patch) const;
    float PatchY(int patch) const;

    // With poison on: which food colour (0 or 1) is poisonous right now.
    int PoisonKind() const;
    int ActivePredators() const;  // the first ActivePredators() entries of `predators` are hunting
    float PredatorSpeed() const;

    Config cfg;
    std::vector<Organism> organisms;
    std::vector<Resource> food;
    std::vector<Resource> bigFood;
    std::vector<Resource> airPockets;
    std::vector<Spring> springs;
    std::vector<Predator> predators;
    std::vector<Slice> slices;
    WorldStats stats;
    float difficulty = 0.0f;  // 0..1, see Config's curriculum settings

private:
    // A remembered parent, used to refill the population after a crash.
    struct Genome {
        NeuralNetwork brain;
        float mutationRate;
        Rgb color;
        int generation;
    };

    void RefillPopulation();
    void AddBodyModules(Organism& o) const;
    void RememberParent(const Organism& parent);
    void SpawnFood();
    void SpawnBigFood();
    void SpawnAirPocket();
    void EatFood(Organism& o);
    void ShareBigFood();
    void MovePredators();
    void UpdateCurriculum();
    Resource RandomResource(float maxX, float radius) const;

    uint64_t nextId_ = 1;
    uint64_t spawnedThisWindow_ = 0;
    int foodCeiling_ = 0;  // the starting maxFood, the most the food curriculum gives back
    std::vector<Genome> archive_;
    size_t archiveNext_ = 0;
};
