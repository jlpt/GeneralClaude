#include "neural_network.h"
#include <random>
#include "raylib.h"
#include <iostream>
#include <string>
#include <omp.h>
bool parallel_enabled = false;



NeuralNetwork::NeuralNetwork() {}

NeuralNetwork::NeuralNetwork(const NeuralNetwork& other)
    : neuron_count(other.neuron_count),
    synapse_count(other.synapse_count),
    outputIndices(other.outputIndices)
{
    neurons.reserve(100000);
    // First, copy the neurons (this makes a shallow copy of each neuron, including its connections)
    neurons = other.neurons;

    // Then, for each neuron, update its connections so that the pointers point to the new neurons.
    // Since we're still using the 24-bit index for destination neuron, we don't need to update pointers here.
    for (size_t i = 0; i < neurons.size(); ++i) {
        for (auto& conn : neurons[i].connections) {
            // Recalculate the destination neuron index if needed
            uint32_t destNeuronIndex = conn.getDestNeuron();
            // Use the new index to update connections if necessary
        }
    }

}

NeuralNetwork::NeuralNetwork(int numNeurons, int numConnections, int numInput, int numOutput)
    : neurons(numNeurons), neuron_count(numNeurons), synapse_count(numConnections)
{
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<> distrib_x(1600, 1700);
    std::uniform_int_distribution<> distrib_y(150, 600);
    std::uniform_int_distribution<> distrib_xin(1500, 1550);
    std::uniform_int_distribution<> distrib_yin(150, 600);
    std::uniform_int_distribution<> distrib_xout(1750, 1800);
    std::uniform_int_distribution<> distrib_yout(150, 600);
    std::uniform_int_distribution<> distrib_index(0, numNeurons - 1);
    std::uniform_int_distribution<> distrib_weight(-100, 100);
    std::uniform_int_distribution<> distrib_refactory(2, 6);

    std::uniform_int_distribution<> distrib_xold(1450, 1500);
    std::uniform_int_distribution<> distrib_yold(150, 702);

    for (int i = 0; i < numNeurons; i++) {
        int rand_x = distrib_x(gen);
        int rand_y = distrib_y(gen);
        int rand_xin = distrib_xin(gen);
        int rand_yin = distrib_yin(gen);
        int rand_xout = distrib_xout(gen);
        int rand_yout = distrib_yout(gen);
        int rand_xold = distrib_xold(gen);
        int rand_yold = distrib_yold(gen);
        std::uniform_real_distribution<> distrib_max(0.75f, 1.0f);
        float max = distrib_max(gen);
        std::uniform_int_distribution<> distrib_min(0.0f, .3f);
        float min = distrib_min(gen);
       // std::cout << max << '\n';
        if (i < numInput) {
            neurons[i] = Neuron(rand_xold - 70, rand_yold, 5, Neuron::NeuronType::Input, 0, 1);
        }
        else if (i >= numNeurons - numOutput) {
            neurons[i] = Neuron(rand_xold + 70, rand_yold, 5, Neuron::NeuronType::Output, 0, 1);
            outputIndices.push_back(i);
        }
        else {
            neurons[i] = Neuron(rand_xold, rand_yold, distrib_refactory(gen), Neuron::NeuronType::Normal, 0, max);
        }
    }

    for (int i = 0; i < numConnections; i++) {
        int source = distrib_index(gen);
        int dest = distrib_index(gen);
        while (neurons[source].type == Neuron::NeuronType::Output) {
            source = distrib_index(gen);
        }
        while (neurons[dest].type == Neuron::NeuronType::Input) {
            dest = distrib_index(gen);
        }

        while (neurons[source].type == Neuron::NeuronType::Input && neurons[dest].type == Neuron::NeuronType::Input ||
            neurons[source].type == Neuron::NeuronType::Input && neurons[dest].type == Neuron::NeuronType::Output) {
            dest = distrib_index(gen);
        }
        if (source == dest)
            dest = (dest + 1) % numNeurons;

        float weight = static_cast<float>(distrib_weight(gen)) / 100;
        // Create a new connection using the 24-bit destination encoding
        neurons[source].connections.emplace_back(weight, static_cast<uint32_t>(dest));
    }
}

void NeuralNetwork::Update() {
    neuron_count = neurons.size();
    size_t total_connections = 0;

    if (parallel_enabled) {
        // Phase 1a: Update activations and count connections in parallel
#pragma omp parallel for reduction(+:total_connections)
        for (int i = 0; i < static_cast<int>(neurons.size()); i++) {
            total_connections += neurons[i].connections.size();
            if (neurons[i].type != Neuron::NeuronType::Output) {
                neurons[i].activation *= 0.98f;
            }
        }
        synapse_count = total_connections;

        // Phase 1b: Propagation and refractory countdown in parallel
#pragma omp parallel for
        for (int i = 0; i < static_cast<int>(neurons.size()); i++) {
            Neuron& n = neurons[i];
            if (n.thresholdReached && n.delay < n.delayTime) {
                n.delay++;
            }
            else if (n.thresholdReached && n.delay >= n.delayTime) {
                // Ensure that Propagate is thread-safe if it accesses shared resources.
                n.Propagate(neurons);
                n.delay = 0;
                n.thresholdReached = false;
            }
            else {
                if (n.refactory > 0) {
                    n.refactory--;
                }
                else {
                    if (n.type != Neuron::NeuronType::Output &&
                        n.activation <= n.thresholdMax &&
                        n.activation >= n.thresholdMin) {
                        n.thresholdReached = true;
                    }
                }
            }
        }

        // Phase 2: Clamp activations between 0 and 1 in parallel
#pragma omp parallel for
        for (int i = 0; i < static_cast<int>(neurons.size()); i++) {
            neurons[i].activation = std::clamp(neurons[i].activation, 0.0f, 1.0f);
        }
    }
    else {
        // Single-threaded version with leaky ReLU and decay
        for (auto& n : neurons) {
            total_connections += n.connections.size();

            // Slowly decay hidden neuron activations
            if (n.type != Neuron::NeuronType::Output) {
                n.activation *= 0.98f;
            }
        }
        synapse_count = total_connections;

        for (int i = 0; i < static_cast<int>(neurons.size()); i++) {
            Neuron& n = neurons[i];

            // Handle delay and propagation
            if (n.thresholdReached && n.delay < n.delayTime) {
                n.delay++;
            }
            else if (n.thresholdReached && n.delay >= n.delayTime) {
                n.Propagate(neurons); // now uses leaky ReLU
                n.delay = 0;
                n.thresholdReached = false;
            }
            else {
                if (n.refactory > 0) {
                    n.refactory--;
                }
                else {
                    // Threshold gating for spike-like firing
                    if (n.type != Neuron::NeuronType::Output &&
                        n.activation >= n.thresholdMin &&
                        n.activation <= n.thresholdMax) {
                        n.thresholdReached = true;
                    }
                }
            }
        }

        // Apply final clamp with slight negative allowance for inhibition
        for (int i = 0; i < static_cast<int>(neurons.size()); i++) {
            neurons[i].activation = std::clamp(neurons[i].activation, -0.1f, 1.0f);
        }
    }
}

// (The rest of the code remains the same, ensuring proper propagation of activations)

void NeuralNetwork::GetSize() {
    size_t totalNeuronMemory = sizeof(Neuron) * neurons.size();
    size_t totalConnectionMemory = 0;

    for (const auto& neuron : neurons)
        totalConnectionMemory += sizeof(Connection) * neuron.connections.capacity();

    size_t totalNetworkMemory = totalNeuronMemory + totalConnectionMemory;

    std::cout << "1 Neuron memory: " << sizeof(Neuron) << " B\n";
    std::cout << "1 Connection memory: " << sizeof(Connection) << " B\n";
    std::cout << "Neuron memory: " << totalNeuronMemory / 1024.0 << " KB\n";
    std::cout << "Connection memory: " << totalConnectionMemory / 1024.0 << " KB\n";
    std::cout << "Total estimated memory: " << totalNetworkMemory / (1024.0 * 1024.0) << " MB\n";
}

void NeuralNetwork::Draw() {
    if (synapse_count <= 5000) {
        for (auto& n : neurons) {
            for (auto& c : n.connections) {
                // Calculate normalized weight for color interpolation
                float normalizedWeight = (c.getWeight() + 1.0f) / 2.0f;
                normalizedWeight = std::clamp(normalizedWeight, 0.0f, 1.0f);
                unsigned char redColor = static_cast<unsigned char>(255 * (1.0f - normalizedWeight));
                unsigned char greenColor = static_cast<unsigned char>(255 * normalizedWeight);

                // Get the destination neuron index and position
                uint32_t destNeuronIndex = c.getDestNeuron();
                if (destNeuronIndex >= neurons.size()) continue;
                Vector2 src = { n.x, n.y };
                Vector2 dst = { neurons[destNeuronIndex].x, neurons[destNeuronIndex].y };

                // Visualize learning changes: if recent weight change exceeds threshold, highlight in yellow
                float delta = std::fabs(c.lastWeightDelta);
                
                if (delta > 0.001f) {
                    // Draw the connection as a thicker yellow line if it's recently changed
                    DrawLineEx(src, dst, 3.0f, YELLOW);
                }
                else {
                    // Otherwise, use the standard weight-based color interpolation
                    DrawLineEx(src, dst, 3.0f, BLACK);
                    DrawLine(n.x, n.y, neurons[destNeuronIndex].x, neurons[destNeuronIndex].y,
                        Color{ redColor, greenColor, 0, 255 });
                }

                // Gradually decay the lastWeightDelta so the highlight fades over time
                c.lastWeightDelta *= 0.9f;
            }
        }
    }

    if (neurons.size() <= 1000) {
        for (size_t i = 0; i < neurons.size(); ++i) {
            Neuron& n = neurons[i];
            unsigned char redColor = static_cast<unsigned char>(255 - static_cast<int>(n.activation * 255));
            unsigned char greenColor = static_cast<unsigned char>(static_cast<int>(n.activation * 255));

            if (n.type == Neuron::NeuronType::Input) {
                DrawCircle(n.x, n.y, 13, BLACK);
                DrawCircle(n.x, n.y, 10, Color{ greenColor, greenColor, greenColor, 255 });

            }
            else if (n.type == Neuron::NeuronType::Output) {
                DrawCircle(n.x, n.y, 13, BLACK);
                DrawCircle(n.x, n.y, 10, Color{ greenColor, 255, redColor, 255 });

            }
            else {
                DrawCircle(n.x, n.y, 13, BLACK);
                DrawCircle(n.x, n.y, 10, Color{ redColor, greenColor, 0, 255 });

            }
        }
    }
}


void NeuralNetwork::Mutate(int power) {
    if (neurons.empty()) return;

    float num = static_cast<float>(neurons.size()) / (11.42f * std::pow(neurons.size(), 1.29f));

    // Original probabilities:
    double addNeuronProbability = 0.01 * num;   // smaller networks: higher chance
    double addWeightProbability = 0.01;           // constant chance

    // New probabilities: removal probabilities are half of the corresponding add probability.
    double removeNeuronProbability = addNeuronProbability / 2;
    double removeWeightProbability = addWeightProbability / 2;

    // The remainder of mutations will modify weights.
    double modifyWeightProbability = 1.0 - (addNeuronProbability + removeNeuronProbability +
        addWeightProbability + removeWeightProbability);

    // Define cumulative probability thresholds.
    double p1 = addNeuronProbability;
    double p2 = p1 + removeNeuronProbability;
    double p3 = p2 + addWeightProbability;
    double p4 = p3 + removeWeightProbability;
    // p5 (modify weight) implicitly is the rest up to 1.0.
   // RemoveNeuron();
    //AddNeuron();
    // 
    //std::cout << "ran" << addNeuronProbability <<'\n';
    //AddNeuron();
    for (int i = 0; i < power; i++) {
        float random = static_cast<float>(rand()) / RAND_MAX;
        if (random <= p1) {
          
          //  AddNeuron();
        }
        else if (random <= p2) {
            //  RemoveNeuron();
        }
        else if (random <= p3) {
          
            AddWeight();
        }
        else if (random <= p4) {
            RemoveWeight();
        }
        else {
            
            ModifyWeight();
        }
    }
    
}

void NeuralNetwork::RemoveWeight() {
    // Remove a randomly chosen connection (weight) from a neuron that has at least one.
    if (neurons.empty()) return;
    std::vector<int> neuronIndices;
    for (size_t i = 0; i < neurons.size(); i++) {
        if (!neurons[i].connections.empty()) {
            neuronIndices.push_back(i);
        }
    }
    if (neuronIndices.empty()) return;  // no connection to remove

    int neuronIndex = neuronIndices[rand() % neuronIndices.size()];
    Neuron& neuron = neurons[neuronIndex];
    int connectionIndex = rand() % neuron.connections.size();
    neuron.connections.erase(neuron.connections.begin() + connectionIndex);
    synapse_count--;
}


// ModifyWeight: Adjust the weight and ensure it stays within the valid range for int8_t using setWeight
void NeuralNetwork::ModifyWeight() {
    int neuron_index = std::rand() % neurons.size();
    Neuron& neuron = neurons[neuron_index];

    if (!neuron.connections.empty()) {
        int connection_index = std::rand() % neuron.connections.size();
        Connection& connection = neuron.connections[connection_index];

        // Generate mutation between -0.1 and 0.1
        float mutation = (static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f) * 0.1f;

        // Get the current weight as a float (in the range [-1, 1])
        float currentWeight = connection.getWeight();

        // Apply the mutation
        float newWeight = currentWeight + mutation;

        // Ensure the new weight stays within the [-1, 1] range
        if (newWeight > 1.0f) {
            newWeight = 1.0f;
        }
        else if (newWeight < -1.0f) {
            newWeight = -1.0f;
        }

        // Set the new weight using setWeight (scales it to int8_t in the range [-127, 127])
        connection.setWeight(newWeight);
    }
}

// AddWeight: Adjust the weight to be within the range of -1.0 to 1.0 and use setWeight
void NeuralNetwork::AddWeight() {
    if (neurons.empty()) {
        return;
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib_index(0, neurons.size() - 1);
    std::uniform_int_distribution<> distrib_weight(-100, 100);  // Mutate weight between -1.0 and 1.0

    int source = distrib_index(gen);
    int dest = distrib_index(gen);

    while (neurons[source].type == Neuron::NeuronType::Output) {
        source = distrib_index(gen);
    }
    while (neurons[dest].type == Neuron::NeuronType::Input) {
        dest = distrib_index(gen);
    }

    if (source == dest)
        dest = (dest + 1) % neurons.size();

    // Generate the weight between -1.0 and 1.0
    float weight = static_cast<float>(distrib_weight(gen)) / 100;

    // Use setWeight to set the weight, automatically scaling it to [-127, 127] range
    neurons[source].connections.emplace_back(weight, static_cast<uint32_t>(dest));
}

// AddNeuron: Ensure weights are properly scaled and within the range of int8_t using setWeight
void NeuralNetwork::AddNeuron() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib_xold(1450, 1500);
    std::uniform_int_distribution<> distrib_yold(150, 702);
    std::uniform_int_distribution<> distrib_refactory(2, 6);

    int rand_x = distrib_xold(gen);
    int rand_y = distrib_yold(gen);
    std::uniform_real_distribution<> distrib_max(0.75f, 1.0f);
    float max = distrib_max(gen);

    Neuron newNeuron(rand_x, rand_y, distrib_refactory(gen), Neuron::NeuronType::Normal, 0, max);
    neurons.push_back(newNeuron);
    int newNeuronIndex = neurons.size() - 1;
    neuron_count++;

    if (newNeuronIndex > 0) {
        std::uniform_int_distribution<> distrib_existing(0, newNeuronIndex - 1);
        int source = distrib_existing(gen);
        while (neurons[source].type == Neuron::NeuronType::Output) {
            source = distrib_existing(gen);
        }

        // Mutate weight between -1.0 and 1.0 for incoming connection
        float weight_in = ((static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f);

        // Use setWeight to set the weight (automatically scales to the int8_t range of [-127, 127])
        neurons[source].connections.push_back(Connection(weight_in, static_cast<uint32_t>(newNeuronIndex)));
        synapse_count++;

        // Mutate weight between -1.0 and 1.0 for outgoing connection
       // float weight_out = ((static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f);

        // Use setWeight to set the weight (automatically scales to the int8_t range of [-127, 127])
        //neurons[newNeuronIndex].connections.push_back(Connection(weight_out, static_cast<uint32_t>(source)));
       // synapse_count++;
    }
}
