// main.cu
#include "raylib.h"
#include <iostream>
#include <vector>
#include <random>
#include "cuda_runtime.h"
#include <stdint.h>    // for int8_t, uint32_t
#include <string>
#include <cmath>
#include <cstdlib>
#ifdef _OPENMP
#include <omp.h>
#endif

// Helper function to format integers with commas.
std::string formatWithCommas(long long value) {
    std::string num = std::to_string(value);
    int insertPosition = num.length() - 3;
    while (insertPosition > 0) {
        num.insert(insertPosition, ",");
        insertPosition -= 3;
    }
    return num;
}

// Simulation configuration.
#define ORGANISM_COUNT 1  // Increase the number of organisms for a larger simulation.
#define NEURON_COUNT 250000
#define CONNECTIONS_PER_NEURON 500
#define TOTAL_NEURONS (ORGANISM_COUNT * NEURON_COUNT)

// Motor neuron configuration (we designate the first 16 neurons of each organism as motor neurons):
#define MOTOR_NEURON_LEFT_START 0
#define MOTOR_NEURON_RIGHT_START 4
#define MOTOR_NEURON_UP_START 8
#define MOTOR_NEURON_DOWN_START 12
#define MOTOR_NEURON_COUNT 4

//------------------------------------------------------------------------------
// Weight-space program: "follow the mouse"
//
// Nothing is trained. The random network is generated exactly as before, then
// a small patch is written straight into its weights so that it steers the
// organism toward the mouse cursor. Press P at runtime to swap between the
// programmed weights and the original random ones.
//
// The circuit reuses neurons 16..20 of each organism as inputs:
//   16 = "target is LEFT"   17 = "target is RIGHT"
//   18 = "target is UP"     19 = "target is DOWN"
//   20 = bias (always 1.0)
// Each frame the host writes how far the target is in each direction
// (0..1, where 1 = SENSOR_RANGE_PX or more) into the four sensor neurons.
//
// Programmed weights:
//   * Every random synapse INTO neurons 0..20 gets weight 0, so the 250k-neuron
//     random reservoir no longer pushes the motors around.
//   * Each sensor neuron's 500 outgoing slots are rewritten: 100 slots at
//     weight +1.0 to each of the 4 motor neurons on its side (gain 100).
//   * The bias neuron gets round(100 * threshold) slots at weight -1.0 onto
//     each motor neuron, one threshold per neuron in the group of 4.
//
// So each frame motor neuron k computes  clamp(a + 100 * (sensor - threshold_k), 0, 1),
// which is about 1 when the target is farther than threshold_k * SENSOR_RANGE_PX
// and 0 otherwise. A motor group therefore outputs 0..4 depending on distance:
// the organism runs at 4 px/frame when far away, slows down as it gets close,
// and stops within MOTOR_THRESHOLDS[0] * SENSOR_RANGE_PX (4 px) of the target.
//
// Besides weights, the program sets refractory periods (also plain data):
// motor neurons never fire (so they hold their value as readouts), and the
// sensor and bias neurons can fire every frame.
//------------------------------------------------------------------------------
#define SENSOR_LEFT 16
#define SENSOR_RIGHT 17
#define SENSOR_UP 18
#define SENSOR_DOWN 19
#define BIAS_NEURON 20
#define INPUT_COUNT 5                 // 4 sensors + bias
#define SPECIAL_NEURON_COUNT 21       // neurons 0..20 of each organism take part in the program
#define SENSOR_RANGE_PX 200.0f
#define SENSOR_GAIN_SYNAPSES 100      // slots from a sensor to each motor neuron it drives
#define MOTOR_NEVER_FIRES 0x7fffffff  // refractory period that will not run out (~1 year at 60 FPS)
static const float MOTOR_THRESHOLDS[MOTOR_NEURON_COUNT] = { 0.02f, 0.10f, 0.25f, 0.50f };  // x SENSOR_RANGE_PX = 4, 20, 50, 100 px

static_assert(SENSOR_GAIN_SYNAPSES * MOTOR_NEURON_COUNT <= CONNECTIONS_PER_NEURON, "a sensor neuron has too few slots for its motor synapses");
static_assert((long long)TOTAL_NEURONS <= (1LL << 24), "packed connections only hold 24-bit destination indices");

// A simple structure to hold neuron drawing info on the CPU (not used in simulation).
struct NeuronCPU {
    int x;
    int y;
};

// A structure to represent an organism for drawing/movement.
struct Organism {
    float circleX;
    float circleY;
};

//------------------------------------------------------------------------------
// New Packed Connection Data Structures
// Each connection is stored in 4 bytes (32 bits):
//   Upper 24 bits: destination neuron index (0 to 16M-1)
//   Lower 8 bits: weight (stored as an int8_t in two's complement, representing values in [-127,127])
//------------------------------------------------------------------------------
struct PackedConnection {
    uint32_t data; // 4 bytes: [dest index (24 bits)] [weight (8 bits)]
};

struct NeuronConnections {
    PackedConnection connections[CONNECTIONS_PER_NEURON];
};

// Helper function to pack a destination index and weight into a 32-bit value.
// The destination should be less than (1 << 24).
inline uint32_t packConnection(uint32_t dest, int8_t weight) {
    return (dest << 8) | (static_cast<uint8_t>(weight));
}

//------------------------------------------------------------------------------
// CUDA Kernels
//------------------------------------------------------------------------------

__global__ void propagateKernel(
    float* d_activations,
    int* d_refactories,
    int* d_startRefactories,
    NeuronConnections* d_neuronConns,
    int totalNeurons
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= totalNeurons) return;

    if (d_refactories[i] <= 0 && d_activations[i] > 0.0f) {
        float sourceAct = d_activations[i];
        // Iterate over the fixed connection array for this neuron.
        for (int j = 0; j < CONNECTIONS_PER_NEURON; ++j) {
            // Unpack the 32-bit value:
            uint32_t packed = d_neuronConns[i].connections[j].data;
            int dest = packed >> 8;                      // Upper 24 bits: destination index.
            int8_t weightByte = static_cast<int8_t>(packed & 0xFF); // Lower 8 bits: weight.
            float weight = weightByte / 127.0f;            // Convert back to [-1, 1] float.
            atomicAdd(&d_activations[dest], sourceAct * weight);
        }
        d_activations[i] = 0.0f;
        d_refactories[i] = d_startRefactories[i];
    }
    else if (d_refactories[i] > 0) {
        d_refactories[i]--;
    }
}

__global__ void clampActivationsKernel(float* d_activations, int totalNeurons) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < totalNeurons) {
        if (d_activations[i] > 1.0f) d_activations[i] = 1.0f;
        if (d_activations[i] < 0.0f) d_activations[i] = 0.0f;
    }
}

__global__ void computeMotorSignalsKernel(const float* d_activations, float* d_motorSignals, int neuronsPerOrganism) {
    int organismIndex = blockIdx.x;  // one block per organism
    int group = threadIdx.x;           // one thread per motor group (0: left, 1: right, 2: up, 3: down)
    if (group < 4) {
        int base = organismIndex * neuronsPerOrganism;
        int start = 0;
        switch (group) {
        case 0: start = MOTOR_NEURON_LEFT_START; break;
        case 1: start = MOTOR_NEURON_RIGHT_START; break;
        case 2: start = MOTOR_NEURON_UP_START; break;
        case 3: start = MOTOR_NEURON_DOWN_START; break;
        }
        float sum = 0.0f;
        for (int i = base + start; i < base + start + MOTOR_NEURON_COUNT; i++) {
            sum += d_activations[i];
        }
        d_motorSignals[organismIndex * 4 + group] = sum;
    }
}

// Writes one set of synapse values (programmed or original) into the listed slots.
// A slot is neuronIndex * CONNECTIONS_PER_NEURON + connectionIndex.
__global__ void applyWeightPatchKernel(
    NeuronConnections* d_neuronConns,
    const unsigned long long* d_slots,
    const uint32_t* d_values,
    int editCount
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= editCount) return;
    unsigned long long slot = d_slots[k];
    d_neuronConns[slot / CONNECTIONS_PER_NEURON].connections[slot % CONNECTIONS_PER_NEURON].data = d_values[k];
}

// Switches neurons 0..20 of each organism between their programmed and original roles.
// One block per organism, one thread per special neuron.
__global__ void setProgramModeKernel(
    float* d_activations,
    int* d_refactories,
    int* d_startRefactories,
    const int* d_originalStartRefactories,  // ORGANISM_COUNT * SPECIAL_NEURON_COUNT
    int programmed,
    int neuronsPerOrganism
) {
    int org = blockIdx.x;
    int local = threadIdx.x;
    if (local >= SPECIAL_NEURON_COUNT) return;
    int i = org * neuronsPerOrganism + local;
    if (programmed) {
        bool isMotor = local < SENSOR_LEFT;
        d_startRefactories[i] = isMotor ? MOTOR_NEVER_FIRES : 0;
        d_refactories[i] = isMotor ? MOTOR_NEVER_FIRES : 0;
        d_activations[i] = 0.0f;
    }
    else {
        d_startRefactories[i] = d_originalStartRefactories[org * SPECIAL_NEURON_COUNT + local];
        d_refactories[i] = 0;
    }
}

// Copies this frame's sensor readings (and the bias) into neurons 16..20 of each organism.
__global__ void injectSensorsKernel(float* d_activations, const float* d_inputs, int neuronsPerOrganism) {
    int org = blockIdx.x;
    int k = threadIdx.x;
    if (k < INPUT_COUNT) {
        d_activations[org * neuronsPerOrganism + SENSOR_LEFT + k] = d_inputs[org * INPUT_COUNT + k];
    }
}

//------------------------------------------------------------------------------
// Builds the weight patch on the host from the freshly generated random weights.
// Only slots whose value actually changes are recorded, together with their
// original value, so the patch can be applied and reverted at runtime.
//------------------------------------------------------------------------------
struct WeightPatch {
    std::vector<unsigned long long> slots;
    std::vector<uint32_t> original;
    std::vector<uint32_t> programmed;
    long long zeroedInbound = 0;    // random synapses into neurons 0..20 that were silenced
    long long rewrittenInputs = 0;  // sensor/bias slots that were rewired
};

WeightPatch buildFollowTargetPatch(const std::vector<NeuronConnections>& conns) {
    WeightPatch patch;
    auto record = [&](unsigned long long neuron, int j, uint32_t newValue, long long& counter) {
        uint32_t oldValue = conns[neuron].connections[j].data;
        if (oldValue == newValue) return;
        patch.slots.push_back(neuron * CONNECTIONS_PER_NEURON + j);
        patch.original.push_back(oldValue);
        patch.programmed.push_back(newValue);
        counter++;
    };

    for (int org = 0; org < ORGANISM_COUNT; org++) {
        unsigned long long base = (unsigned long long)org * NEURON_COUNT;
        for (int local = 0; local < NEURON_COUNT; local++) {
            unsigned long long neuron = base + local;
            bool isInput = local >= SENSOR_LEFT && local < SENSOR_LEFT + INPUT_COUNT;

            if (!isInput) {
                // Silence every synapse that lands on a motor, sensor or bias neuron.
                for (int j = 0; j < CONNECTIONS_PER_NEURON; j++) {
                    uint32_t data = conns[neuron].connections[j].data;
                    uint32_t destLocal = (data >> 8) - (uint32_t)base;
                    if (destLocal < SPECIAL_NEURON_COUNT) {
                        record(neuron, j, data & 0xFFFFFF00u, patch.zeroedInbound);
                    }
                }
                continue;
            }

            // Rewire the input neuron's 500 slots. Unused slots keep their destination, weight 0.
            std::vector<uint32_t> row(CONNECTIONS_PER_NEURON);
            for (int j = 0; j < CONNECTIONS_PER_NEURON; j++) {
                row[j] = conns[neuron].connections[j].data & 0xFFFFFF00u;
            }
            int slot = 0;
            if (local == BIAS_NEURON) {
                // Inhibit motor neuron k of every group by round(100 * threshold_k).
                const int groupStarts[4] = { MOTOR_NEURON_LEFT_START, MOTOR_NEURON_RIGHT_START, MOTOR_NEURON_UP_START, MOTOR_NEURON_DOWN_START };
                int needed = 0;
                for (int k = 0; k < MOTOR_NEURON_COUNT; k++) needed += 4 * (int)(SENSOR_GAIN_SYNAPSES * MOTOR_THRESHOLDS[k] + 0.5f);
                if (needed > CONNECTIONS_PER_NEURON) {
                    std::cerr << "Bias neuron needs " << needed << " slots but only has " << CONNECTIONS_PER_NEURON << std::endl;
                    std::exit(1);
                }
                for (int g = 0; g < 4; g++) {
                    for (int k = 0; k < MOTOR_NEURON_COUNT; k++) {
                        int count = (int)(SENSOR_GAIN_SYNAPSES * MOTOR_THRESHOLDS[k] + 0.5f);
                        for (int c = 0; c < count; c++) {
                            row[slot++] = packConnection((uint32_t)(base + groupStarts[g] + k), -127);
                        }
                    }
                }
            }
            else {
                // Excite the 4 motor neurons on this sensor's side, SENSOR_GAIN_SYNAPSES slots each.
                const int sideStart[4] = { MOTOR_NEURON_LEFT_START, MOTOR_NEURON_RIGHT_START, MOTOR_NEURON_UP_START, MOTOR_NEURON_DOWN_START };
                int motorStart = sideStart[local - SENSOR_LEFT];
                for (int k = 0; k < MOTOR_NEURON_COUNT; k++) {
                    for (int c = 0; c < SENSOR_GAIN_SYNAPSES; c++) {
                        row[slot++] = packConnection((uint32_t)(base + motorStart + k), 127);
                    }
                }
            }
            for (int j = 0; j < CONNECTIONS_PER_NEURON; j++) {
                record(neuron, j, row[j], patch.rewrittenInputs);
            }
        }
    }
    return patch;
}

//------------------------------------------------------------------------------
// Main Function
//------------------------------------------------------------------------------

int main() {
    // Set up random generators.
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distribRefactory(1,60);
    std::uniform_int_distribution<> distribX(50, 750);
    std::uniform_int_distribution<> distribY(50, 550);
    // For connection destinations (each organism's network is isolated).
    std::uniform_int_distribution<> distribLocalNeuron(0, NEURON_COUNT - 1);
    std::uniform_real_distribution<float> distribWeight(-1.0f, 1.0f);

    // Host arrays.
    std::vector<NeuronCPU> neurons(TOTAL_NEURONS);
    std::vector<float> activations(TOTAL_NEURONS, 0.0f);
    std::vector<int> refactories(TOTAL_NEURONS, 0);
    std::vector<int> startRefactories(TOTAL_NEURONS, 0);

    // Initialize refractory times.
    for (int i = 0; i < TOTAL_NEURONS; i++) {
        startRefactories[i] = distribRefactory(gen);
    }

    // Create per-neuron connection arrays.
    std::vector<NeuronConnections> neuronConns(TOTAL_NEURONS);

    // Initialize neurons, their activations, and their connections.
    for (int org = 0; org < ORGANISM_COUNT; org++) {
        int baseNeuron = org * NEURON_COUNT;
        for (int i = 0; i < NEURON_COUNT; i++) {
            int neuronIndex = baseNeuron + i;
            neurons[neuronIndex].x = distribX(gen);
            neurons[neuronIndex].y = distribY(gen);
            activations[neuronIndex] = 1.0f;  // start active

            // For each connection for this neuron, pack the destination and weight.
            for (int j = 0; j < CONNECTIONS_PER_NEURON; j++) {
                // Destination is within the same organism.
                uint32_t dest = baseNeuron + distribLocalNeuron(gen);
                float w = distribWeight(gen);
                int8_t weight = static_cast<int8_t>(w * 127);
                neuronConns[neuronIndex].connections[j].data = packConnection(dest, weight);
            }
        }
    }

    // Write the "follow the mouse" program into the weights (see the comment at the top).
    WeightPatch patch = buildFollowTargetPatch(neuronConns);
    const int editCount = (int)patch.slots.size();
    const long long totalSynapses = (long long)TOTAL_NEURONS * CONNECTIONS_PER_NEURON;
    const double editedPercent = 100.0 * editCount / (double)totalSynapses;
    std::cout << "Weight patch: " << formatWithCommas(editCount) << " of " << formatWithCommas(totalSynapses)
        << " synapses edited (" << editedPercent << "%): " << formatWithCommas(patch.zeroedInbound)
        << " random synapses into neurons 0-" << (SPECIAL_NEURON_COUNT - 1) << " silenced, "
        << formatWithCommas(patch.rewrittenInputs) << " sensor/bias slots rewired." << std::endl;

    std::vector<int> originalSpecialStart(ORGANISM_COUNT * SPECIAL_NEURON_COUNT);
    for (int org = 0; org < ORGANISM_COUNT; org++) {
        for (int k = 0; k < SPECIAL_NEURON_COUNT; k++) {
            originalSpecialStart[org * SPECIAL_NEURON_COUNT + k] = startRefactories[org * NEURON_COUNT + k];
        }
    }

    // Device pointers.
    float* d_activations;
    int* d_refactories;
    int* d_startRefactories;
    NeuronConnections* d_neuronConns;
    float* d_motorSignals;  // one set of 4 signals per organism

    // Allocate device memory.
    cudaMalloc((void**)&d_activations, TOTAL_NEURONS * sizeof(float));
    cudaMalloc((void**)&d_refactories, TOTAL_NEURONS * sizeof(int));
    cudaMalloc((void**)&d_startRefactories, TOTAL_NEURONS * sizeof(int));
    cudaMalloc((void**)&d_neuronConns, TOTAL_NEURONS * sizeof(NeuronConnections));
    cudaMalloc((void**)&d_motorSignals, ORGANISM_COUNT * 4 * sizeof(float));

    // Copy host data to device.
    cudaMemcpy(d_activations, activations.data(), TOTAL_NEURONS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_refactories, refactories.data(), TOTAL_NEURONS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_startRefactories, startRefactories.data(), TOTAL_NEURONS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_neuronConns, neuronConns.data(), TOTAL_NEURONS * sizeof(NeuronConnections), cudaMemcpyHostToDevice);

    // Device copies of the weight patch, the special neurons' original refractory
    // periods, and the per-frame sensor inputs.
    unsigned long long* d_patchSlots;
    uint32_t* d_patchProgrammed;
    uint32_t* d_patchOriginal;
    int* d_originalSpecialStart;
    float* d_inputs;
    cudaMalloc((void**)&d_patchSlots, editCount * sizeof(unsigned long long));
    cudaMalloc((void**)&d_patchProgrammed, editCount * sizeof(uint32_t));
    cudaMalloc((void**)&d_patchOriginal, editCount * sizeof(uint32_t));
    cudaMalloc((void**)&d_originalSpecialStart, ORGANISM_COUNT * SPECIAL_NEURON_COUNT * sizeof(int));
    cudaMalloc((void**)&d_inputs, ORGANISM_COUNT * INPUT_COUNT * sizeof(float));
    cudaMemcpy(d_patchSlots, patch.slots.data(), editCount * sizeof(unsigned long long), cudaMemcpyHostToDevice);
    cudaMemcpy(d_patchProgrammed, patch.programmed.data(), editCount * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_patchOriginal, patch.original.data(), editCount * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_originalSpecialStart, originalSpecialStart.data(), ORGANISM_COUNT * SPECIAL_NEURON_COUNT * sizeof(int), cudaMemcpyHostToDevice);
    std::vector<float> inputs(ORGANISM_COUNT * INPUT_COUNT, 0.0f);

    // Switch between the programmed and the original random weights.
    auto setProgrammed = [&](bool on) {
        int patchBlocks = (editCount + 255) / 256;
        if (patchBlocks > 0) {
            applyWeightPatchKernel << <patchBlocks, 256 >> > (d_neuronConns, d_patchSlots, on ? d_patchProgrammed : d_patchOriginal, editCount);
        }
        setProgramModeKernel << <ORGANISM_COUNT, SPECIAL_NEURON_COUNT >> > (d_activations, d_refactories, d_startRefactories, d_originalSpecialStart, on ? 1 : 0, NEURON_COUNT);
        cudaDeviceSynchronize();
    };
    bool programmed = true;
    setProgrammed(programmed);

    // Initialize Raylib window.
    InitWindow(800, 600, "CUDA Neural Network Simulation (Packed Connections)");
    SetTargetFPS(60);

    // Create organisms for drawing.
    std::vector<Organism> organisms(ORGANISM_COUNT);
    for (int i = 0; i < ORGANISM_COUNT; i++) {
        organisms[i].circleX = 400.0f;
        organisms[i].circleY = 300.0f;
    }
    const float speedFactor = 1.0f;

    // Setup grid for launching kernels.
    int threadsPerBlock = 256;
    int blocksPerGrid = (TOTAL_NEURONS + threadsPerBlock - 1) / threadsPerBlock;

    // Main simulation loop.
    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) {
            programmed = !programmed;
            setProgrammed(programmed);
        }

        // Sense: how far is the target (the mouse) to the left/right/up/down of each organism?
        if (programmed) {
            Vector2 target = GetMousePosition();
            for (int org = 0; org < ORGANISM_COUNT; org++) {
                float dx = (target.x - organisms[org].circleX) / SENSOR_RANGE_PX;
                float dy = (target.y - organisms[org].circleY) / SENSOR_RANGE_PX;
                float* in = &inputs[org * INPUT_COUNT];
                in[0] = fminf(fmaxf(-dx, 0.0f), 1.0f);  // target is left
                in[1] = fminf(fmaxf(dx, 0.0f), 1.0f);   // target is right
                in[2] = fminf(fmaxf(-dy, 0.0f), 1.0f);  // target is up (screen y grows downward)
                in[3] = fminf(fmaxf(dy, 0.0f), 1.0f);   // target is down
                in[4] = 1.0f;                           // bias
            }
            cudaMemcpy(d_inputs, inputs.data(), ORGANISM_COUNT * INPUT_COUNT * sizeof(float), cudaMemcpyHostToDevice);
            injectSensorsKernel << <ORGANISM_COUNT, INPUT_COUNT >> > (d_activations, d_inputs, NEURON_COUNT);
        }

        // Propagate neural activations.
        propagateKernel << <blocksPerGrid, threadsPerBlock >> > (d_activations, d_refactories, d_startRefactories, d_neuronConns, TOTAL_NEURONS);
        cudaDeviceSynchronize();

        // Clamp activations.
        clampActivationsKernel << <blocksPerGrid, threadsPerBlock >> > (d_activations, TOTAL_NEURONS);
        cudaDeviceSynchronize();

        // Compute motor signals for each organism.
        computeMotorSignalsKernel << <ORGANISM_COUNT, 4 >> > (d_activations, d_motorSignals, NEURON_COUNT);
        cudaDeviceSynchronize();

        // Copy motor signals back to host.
        float motorSignals[ORGANISM_COUNT * 4];
        cudaMemcpy(motorSignals, d_motorSignals, ORGANISM_COUNT * 4 * sizeof(float), cudaMemcpyDeviceToHost);

        // Update organism positions in parallel using OpenMP.
#pragma omp parallel for
        for (int org = 0; org < ORGANISM_COUNT; org++) {
            float leftSignal = motorSignals[org * 4 + 0];
            float rightSignal = motorSignals[org * 4 + 1];
            float upSignal = motorSignals[org * 4 + 2];
            float downSignal = motorSignals[org * 4 + 3];
            organisms[org].circleX += (rightSignal - leftSignal) * speedFactor;
            organisms[org].circleY += (downSignal - upSignal) * speedFactor;
            if (organisms[org].circleX < 0) organisms[org].circleX = 0;
            if (organisms[org].circleX > 800) organisms[org].circleX = 800;
            if (organisms[org].circleY < 0) organisms[org].circleY = 0;
            if (organisms[org].circleY > 600) organisms[org].circleY = 600;
        }

        // Begin drawing.
        BeginDrawing();
        ClearBackground(SKYBLUE);
        // Use 64-bit math for synapse count display.
        std::string simInfo = "Simulating " + formatWithCommas(ORGANISM_COUNT) + " organisms (" +
            formatWithCommas(TOTAL_NEURONS) + " neurons, " +
            formatWithCommas((long long)TOTAL_NEURONS * CONNECTIONS_PER_NEURON) + " synapses)";
        DrawText(simInfo.c_str(), 10, 10, 20, WHITE);
        DrawFPS(10, 70);

        Vector2 mousePos = GetMousePosition();
        if (programmed) {
            DrawText(TextFormat("PROGRAMMED weights: follows the mouse (%s synapses edited, %.4f%%)  [P] original",
                formatWithCommas(editCount).c_str(), editedPercent), 10, 40, 20, YELLOW);
            DrawCircleLines(static_cast<int>(mousePos.x), static_cast<int>(mousePos.y), 6, RED);
        }
        else {
            DrawText("ORIGINAL random weights  [P] programmed", 10, 40, 20, WHITE);
        }

        // Display organism info when hovering.
        for (int org = 0; org < ORGANISM_COUNT; org++) {

            DrawCircle(static_cast<int>(organisms[org].circleX), static_cast<int>(organisms[org].circleY), 16, WHITE);
            if (CheckCollisionPointCircle(mousePos, Vector2{ organisms[org].circleX, organisms[org].circleY }, 16)) {
                int neuronCountVal = NEURON_COUNT;
                long long synapseCountVal = (long long)NEURON_COUNT * CONNECTIONS_PER_NEURON;

                // Determine neuron rank based on thresholds.
                const char* neuronRank =
                    (neuronCountVal < 10) ? "Spark" :
                    (neuronCountVal < 25) ? "Cell" :
                    (neuronCountVal < 50) ? "Node" :
                    (neuronCountVal < 100) ? "Cluster" :
                    (neuronCountVal < 250) ? "Core" :
                    (neuronCountVal < 500) ? "Lobe" :
                    (neuronCountVal < 1000) ? "Complex" :
                    (neuronCountVal < 2500) ? "System" :
                    (neuronCountVal < 5000) ? "Cerebrum" :
                    (neuronCountVal < 10000) ? "Cortex" :
                    (neuronCountVal < 25000) ? "Overmind" :
                    (neuronCountVal < 50000) ? "Omnipresence" :
                    (neuronCountVal < 100000) ? "Transcendence" :
                    (neuronCountVal < 250000) ? "Ascendancy" :
                    (neuronCountVal < 500000) ? "Eternity" :
                    (neuronCountVal < 1000000) ? "Supremacy" :
                    (neuronCountVal < 2000000) ? "Quantum Consciousness" :
                    (neuronCountVal < 3000000) ? "Cerebral Cortex" :
                    (neuronCountVal < 4000000) ? "Cosmic" :
                    (neuronCountVal < 5000000) ? "Galactic" :
                    "Universe";

                // Determine connection rank based on thresholds.
                const char* connectionRank =
                    (synapseCountVal < 100) ? "Sparse" :
                    (synapseCountVal < 500) ? "Linked" :
                    (synapseCountVal < 1000) ? "Chained" :
                    (synapseCountVal < 2500) ? "Integrated" :
                    (synapseCountVal < 5000) ? "Webbed" :
                    (synapseCountVal < 10000) ? "Latticed" :
                    (synapseCountVal < 50000) ? "Matrixed" :
                    (synapseCountVal < 100000) ? "Neurogrid" :
                    (synapseCountVal < 250000) ? "Hyperlinked" :
                    (synapseCountVal < 1000000) ? "Quantum Matrix" :
                    (synapseCountVal < 5000000) ? "Metalinked" :
                    (synapseCountVal < 10000000) ? "Entangled" :
                    (synapseCountVal < 25000000) ? "Interwoven" :
                    (synapseCountVal < 50000000) ? "Quantum Webbed" :
                    (synapseCountVal < 100000000) ? "Synaptic Storm" :
                    (synapseCountVal < 250000000) ? "Digital Nexus" :
                    (synapseCountVal < 500000000) ? "Cybernetic Web" :
                    (synapseCountVal < 1000000000) ? "Neural Empire" :
                    "Abyss of Connections";

                // Choose a color based on the synapse count.
                Color rankColor =
                    (synapseCountVal < 250) ? WHITE :
                    (synapseCountVal < 500) ? BLUE :
                    (synapseCountVal < 1000) ? GREEN :
                    (synapseCountVal < 2500) ? DARKGREEN :
                    (synapseCountVal < 5000) ? DARKBLUE :
                    (synapseCountVal < 10000) ? PURPLE :
                    (synapseCountVal < 25000) ? GOLD :
                    (synapseCountVal < 50000) ? ORANGE :
                    (synapseCountVal < 100000) ? MAROON :
                    (synapseCountVal < 250000) ? LIME :
                    (synapseCountVal < 500000) ? DARKGRAY :
                    (synapseCountVal < 1000000) ? VIOLET :
                    (synapseCountVal < 2500000) ? MAGENTA :
                    (synapseCountVal < 5000000) ? YELLOW :
                    (synapseCountVal < 10000000) ? RED :
                    (synapseCountVal < 25000000) ? PINK :
                    (synapseCountVal < 50000000) ? Color{ 23,81,222,255 } :
                    (synapseCountVal < 100000000) ? BEIGE :
                    (synapseCountVal < 250000000) ? BROWN :
                    (synapseCountVal < 500000000) ? DARKBROWN :
                    (synapseCountVal < 1000000000) ? DARKPURPLE :
                    BLACK;

                DrawText(TextFormat("%s-%s", connectionRank, neuronRank),
                    GetMousePosition().x + 16, GetMousePosition().y, 20, rankColor);

                DrawText(TextFormat("Neurons: %s", formatWithCommas(NEURON_COUNT).c_str()),
                    mousePos.x + 16, mousePos.y + 32, 20, WHITE);
                DrawText(TextFormat("Synapses: %s", formatWithCommas((long long)NEURON_COUNT * CONNECTIONS_PER_NEURON).c_str()),
                    mousePos.x + 16, mousePos.y + 16, 20, WHITE);
            }
        }
        EndDrawing();
    }

    // Cleanup.
    CloseWindow();
    cudaFree(d_activations);
    cudaFree(d_refactories);
    cudaFree(d_startRefactories);
    cudaFree(d_neuronConns);
    cudaFree(d_motorSignals);
    cudaFree(d_patchSlots);
    cudaFree(d_patchProgrammed);
    cudaFree(d_patchOriginal);
    cudaFree(d_originalSpecialStart);
    cudaFree(d_inputs);

    return 0;
}
