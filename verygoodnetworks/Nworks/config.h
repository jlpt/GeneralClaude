#pragma once
#include <cstdint>

// Every tunable number in the simulation lives here. The defaults reproduce
// the original setup.
struct Config {
    // --- World ---
    float worldWidth = 1300.0f;
    float worldHeight = 768.0f;

    // Area where random organisms, food and the first air pockets appear.
    float spawnMinX = 40.0f;
    float spawnMaxX = 700.0f;
    float spawnMinY = 80.0f;
    float spawnMaxY = 702.0f;
    float airSpawnMaxX = 300.0f;  // later air pockets start in [spawnMinX, airSpawnMaxX]

    // --- Population ---
    int minPopulation = 15;            // below this, one random organism spawns per step
    float organismRadius = 12.0f;
    float maxHealth = 5000.0f;         // steps an organism lasts without eating food
    float maxAir = 10000.0f;           // steps an organism lasts without an air pocket
    int foodToReproduce = 3;
    float moveSpeed = 3.0f;
    float offspringSpread = 20.0f;     // children appear up to this far from the parent

    // --- Brains ---
    int initialHiddenNeurons = 43;
    int initialConnections = 600;
    float brainCostPerSynapse = 0.0f;  // extra health lost per step per synapse (0 = brains are free)

    // --- Leaky integrate-and-fire neurons ---
    float membraneLeak = 0.95f;        // potential is multiplied by this every step
    float minThreshold = 0.5f;         // hidden-neuron firing thresholds stay in this range
    float maxThreshold = 1.0f;
    int minRefractory = 2;             // steps a neuron stays silent after it spikes
    int maxRefractory = 6;
    float outputDecay = 0.9f;          // outputs smooth incoming spikes into a 0..1 level

    // --- Mutation ---
    float mutationFraction = 0.02f;        // mutations per birth = synapses * fraction * mutationRate
    float macroMutationChance = 0.004f;    // 1 in 250 births mutate much harder...
    float macroMutationMultiplier = 15.0f; // ...this many times harder
    float mutationRateDrift = 0.01f;       // child's mutation rate = parent's * (1 +/- drift)
    float weightStep = 0.1f;               // largest change from one weight mutation
    float thresholdStep = 0.05f;           // largest change from one threshold mutation
    // Chance that a single mutation is each of these; anything else tweaks a weight.
    float addNeuronChance = 0.002f;
    float removeNeuronChance = 0.001f;
    float addConnectionChance = 0.01f;
    float removeConnectionChance = 0.005f;
    float thresholdChance = 0.02f;

    // --- Resources ---
    int initialFood = 100;
    int maxFood = 100;
    float foodSpawnChance = 0.015f;    // per step
    float foodRadius = 5.0f;
    int initialAirPockets = 10;
    int maxAirPockets = 50;
    float airSpawnChance = 0.015f;     // per step
    float airPocketRadius = 8.0f;

    // Curriculum: as the leading generation grows, new food spawns further
    // left and new air pockets further right, so organisms must travel
    // between them (eventually beyond the range of their senses).
    float foodDriftPerGeneration = 0.5f;
    float foodDriftMax = 400.0f;
    float airDriftPerGeneration = 5.0f;
    float airDriftMax = 1000.0f;

    // --- Senses ---
    float resourceSensorRange = 500.0f;
    float organismSensorRange = 600.0f;
    float sliceSensorRange = 600.0f;

    // --- Optional features (off in the original setup) ---
    bool organismSensor = false;  // sense the nearest other organism (+5 inputs)
    bool positionSensor = false;  // sense own x/y position (+2 inputs)
    bool slices = false;          // organisms can cut lines that kill (+8 inputs, +4 outputs)
    int sliceCooldown = 10;       // steps between cuts

    uint32_t seed = 0;            // 0 = pick a random seed
};
