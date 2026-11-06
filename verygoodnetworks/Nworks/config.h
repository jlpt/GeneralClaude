#pragma once
#include <cstdint>

// Every tunable number in the simulation lives here.
//
// To get the original world back, set brainCostPerNeuron and
// brainCostPerSynapse to 0, turn off vision, springs and seasons, and turn
// off everything in stage 4.
struct Config {
    // --- World ---
    float worldWidth = 1300.0f;
    float worldHeight = 768.0f;

    // Area where random organisms appear, and where food and air pockets
    // appear when seasons/springs are off.
    float spawnMinX = 40.0f;
    float spawnMaxX = 700.0f;
    float spawnMinY = 80.0f;
    float spawnMaxY = 702.0f;
    float airSpawnMaxX = 300.0f;  // later air pockets start in [spawnMinX, airSpawnMaxX]

    // --- Population ---
    int minPopulation = 15;            // below this, one organism is added per step
    // Refill the population mostly with mutated copies of recent parents (a
    // "seed bank") instead of random brains, so a crash doesn't throw away
    // everything evolution has found.
    bool refillFromSurvivors = true;
    int survivorArchiveSize = 50;      // recent parents remembered
    float randomRefillShare = 0.2f;    // fraction of refills that are brand-new random brains
    float organismRadius = 12.0f;
    float maxHealth = 5000.0f;         // steps an organism lasts without eating (before brain cost)
    float maxAir = 10000.0f;           // steps an organism lasts without air
    int foodToReproduce = 3;
    int maxAge = 0;                    // steps; 0 = organisms don't age
    float moveSpeed = 3.0f;
    float offspringSpread = 20.0f;     // children appear up to this far from the parent

    // --- Brains ---
    int initialHiddenNeurons = 43;
    int initialConnections = 600;
    // Brains burn energy: every step, health drops by an extra
    // neurons * brainCostPerNeuron + synapses * brainCostPerSynapse.
    // The starting brain costs about 25% extra; a brain has to earn its size.
    float brainCostPerNeuron = 0.002f;
    float brainCostPerSynapse = 0.0002f;

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

    // --- Food ---
    int initialFood = 100;
    int maxFood = 100;
    float foodSpawnChance = 0.03f;     // per step (the original was 0.015, before brains cost energy)
    float foodRadius = 5.0f;

    // --- Air pockets (used when springs are off) ---
    int initialAirPockets = 10;
    int maxAirPockets = 50;
    float airSpawnChance = 0.015f;     // per step
    float airPocketRadius = 8.0f;

    // --- Curriculum ---
    // Difficulty (0..1) rises while the population sustains itself (no
    // random spawns needed) and falls back when it collapses.
    int curriculumWindow = 20000;      // steps between difficulty updates
    float curriculumStep = 0.025f;     // change per update
    int curriculumBackoffSpawns = 10;  // random spawns in one window that count as a collapse
    // With seasons off: food shifts left and air pockets right by difficulty * these.
    float foodDriftMax = 400.0f;
    float airDriftMax = 1000.0f;

    // --- Stage 2: vision ---
    // The view around an organism is split into `visionSectors` slices. For
    // each slice, three inputs report how close the nearest food, air source
    // and other organism are (1 = touching, 0 = nothing within visionRange).
    bool vision = true;                // false = the original nearest-food/air sensors
    int visionSectors = 8;             // up to 32
    float visionRange = 300.0f;
    float resourceSensorRange = 500.0f;  // range of the original sensors

    // --- Stage 3: springs and seasons ---
    // Springs: three fixed air sources, further apart than organisms can see.
    // Each holds a limited amount of air and refills slowly, so a crowd can
    // drain one and has to remember where the others are.
    bool springs = true;               // replaces air pockets
    float springRadius = 18.0f;
    float springCapacity = 30000.0f;   // air stored when full
    float springRecharge = 15.0f;      // air regained per step

    // Seasons: food blooms in one of three patches, and the patch moves on
    // every `seasonLength` steps. Uneaten food rots. Difficulty sets how much
    // food grows in the patch versus scattered anywhere (0 = all scattered).
    bool seasons = true;
    int seasonLength = 8000;
    float patchRadius = 150.0f;
    int foodLifetime = 6000;           // steps before uneaten food rots (seasons only)

    // --- Stage 4a: signals ---
    // One output broadcasts a 0..1 signal (it costs energy to call); four
    // inputs hear the loudest nearby call from the left, up, right and down.
    bool signals = true;
    float hearingRange = 400.0f;
    float signalCost = 0.1f;           // extra health lost per step at full volume

    // --- Stage 4b: cooperative food ---
    // Big food (gold) is worth bigFoodValue food each, but only opens when
    // at least bigFoodPartners organisms touch it at the same time.
    bool cooperativeFood = true;
    int maxBigFood = 4;
    float bigFoodSpawnChance = 0.002f;
    float bigFoodRadius = 16.0f;
    int bigFoodPartners = 2;
    int bigFoodValue = 2;

    // --- Stage 4c: predators ---
    // Predators chase the nearest organism they can see and kill on contact.
    // They arrive with the curriculum (the first at 25% difficulty, the
    // second at 75%) and get faster as it gets harder.
    bool predators = true;
    int predatorCount = 2;             // at 100% difficulty
    float predatorRadius = 14.0f;
    float predatorSight = 200.0f;
    float predatorBaseSpeed = 0.8f;
    float predatorExtraSpeed = 2.2f;   // added at 100% difficulty (organisms top out at 3 per axis)
    int predatorRest = 1000;           // steps a predator rests after a catch

    // --- Stage 4d: poison lookalikes ---
    // Food comes in two colours. One is poisonous, and which one swaps every
    // poisonSwapInterval steps: about once per lifetime, too often for
    // evolution to hard-wire, so it pays to learn. The lookalikes are extra:
    // food spawns twice as often (and up to twice maxFood), so there is as
    // much good food as without poison.
    bool poison = true;
    int poisonSwapInterval = 10000;
    float poisonDamage = 0.4f;         // fraction of maxHealth lost per poisonous bite at 100% difficulty
                                       // (it scales with difficulty; bad bites always taste bad)
    bool poisonScalesWithDifficulty = true;  // false: every poisonous bite costs the full poisonDamage

    // --- Stage 4e: lifetime learning ---
    // Synapses change during life by a reward-modulated Hebbian rule: a
    // synapse whose spike helped its target fire becomes "eligible", and
    // eligible synapses strengthen after a reward (good food) or weaken after
    // a punishment (poison). Each brain's learning rate is a gene; what it
    // learns is not inherited.
    bool plasticity = true;
    bool learningHasEffect = true;
    bool learnFromReward = true;       // false: good food doesn't drive learning (experiments)
    bool learnFromPunishment = true;   // false: poison doesn't drive learning (experiments)     // false: the learning-rate gene still mutates but nothing is learned (neutral control)
    float traceDecay = 0.9f;           // how long a pairing stays eligible (per step)
    float learnedForgetting = 0.0002f; // learned changes fade by this fraction per step (half-life ~3,500 steps)
    float rewardDecay = 0.9f;          // the reward/punishment signal fades by this per step
    float maxInitialLearningRate = 0.05f;
    float learningRateStep = 0.01f;    // largest change from one learning-rate mutation
    float learningRateChance = 0.02f;  // chance a mutation changes the learning rate

    // --- Optional features ---
    bool organismSensor = false;  // original nearest-organism sensor (+5 inputs); vision already sees organisms
    bool positionSensor = false;  // sense own x/y position (+2 inputs)
    bool slices = false;          // organisms can cut lines that kill (+8 inputs, +4 outputs)
    bool colourSmell = false;     // +2 inputs: closeness of the nearest red and nearest purple food, from any direction
    int sliceCooldown = 10;       // steps between cuts
    float sliceSensorRange = 600.0f;
    float organismSensorRange = 600.0f;

    uint32_t seed = 0;            // 0 = pick a random seed
};
