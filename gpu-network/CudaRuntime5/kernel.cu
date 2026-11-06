// main.cu
#include "raylib.h"
#include <iostream>
#include <vector>
#include <random>
#include "cuda_runtime.h"
#include <stdint.h>    // for int8_t, uint32_t
#include <string>
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

        // Display organism info when hovering.
        Vector2 mousePos = GetMousePosition();
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

    return 0;
}
