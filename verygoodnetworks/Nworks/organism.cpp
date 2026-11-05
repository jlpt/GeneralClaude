#include "organism.h"
#include "neural_network.h"
#include "raylib.h"
#include <iostream>
#include <cstdlib>   // For rand()
#include <ctime>     // For time()

Organism::Organism(float x, float y, int size)
    : x(x), y(y), size(size) {
    health = 5000.0f;
    air = 10000.0f;
    color = Color{
        static_cast<unsigned char>(rand() % 256),
        static_cast<unsigned char>(rand() % 256),
        static_cast<unsigned char>(rand() % 256),
        255  // Fully opaque
    };
}


void Organism::InitNetwork(int neurons, int connections, int inputs, int outputs)
{
    input = inputs;
    output = outputs;
    network = std::make_unique<NeuralNetwork>(neurons, connections, inputs, outputs);
    
}

void Organism::AddModule(std::unique_ptr<Module> module) {
    // First, calculate the inputs and outputs already used by existing modules.
    int usedInputs = 0;
    int usedOutputs = 0;
    for (const auto& m : modules) {
        
        usedInputs += m->input;   // assumes each module has an 'input' member
        usedOutputs += m->output;   // and an 'output' member
    }
    //module->SetIndexOffsets(usedInputs, usedOutputs);
    // Check if adding the new module would exceed capacity.
    if (usedInputs + module->input <= input && usedOutputs + module->output <= output) {
        
     
        
        inputsUsed = usedInputs+module->input;
        outputsUsed = usedOutputs+module->output;
       // std::cout << module.get()->inputStartingNeuron << '\n';
        modules.push_back(std::move(module));
       
       // std::cout << "Module added successfully." << std::endl;
        
        
    } else {
        std::cerr << "Error: Not enough available inputs/outputs to add module." << std::endl;
        // Optionally, you might delete or handle the module differently here.
    }
}

void Organism::Update() {
    if (!alive) {
        //return;
    }
    if (health <= 0) {
        alive = false; // Mark as dead
       // std::cout << "Organism at (" << x << ", " << y << ") has died!\n";
        //return; // Stop further updates for dead organism
    }
    if (air <= 0) {
        alive = false;
    }
    // Update all sensors in the list.
    health = health - 1;//*((float)network->synapse_count/100.0f);
    air = air - 1;
    network->Update();
    for (auto& module : modules) {
        module->Update();
    }
    if (x < 0) {
        x = 0;
    }
    if (x >= 1300) {
        x = 1300;
    }
    if (y >= 768) {
        y = 768;
    }
    if (y < 6) {
        y = 6;
    }

}

void Organism::Draw() {
    DrawCircle(x, y, size + 3, BLACK);
    DrawCircle(x, y, size, color);
}

void Organism::DrawNetwork() {
    network->Draw();
}

void Organism::Click() {
    std::cout << "Organism clicked at position (" << x << ", " << y << ")" << std::endl;
}

std::unique_ptr<Organism> Organism::Reproduce() {
    int random = rand();
   

        float offsetX = static_cast<float>((rand() % 40) - 20); // +/- 20
        float offsetY = static_cast<float>((rand() % 40) - 20);
    

    auto offspring = std::make_unique<Organism>(x + offsetX, y + offsetY, size);

    // Deep copy the neural network (assuming NeuralNetwork has a deep copy constructor)
    offspring->network = std::make_unique<NeuralNetwork>(*network);
    // Optionally, mutate the offspring's network:
   // std::cout << (int)(network->synapse_count / (10 / mutationRate)) << '\n';
    //offspring->network->Mutate((int)(network->synapse_count/(10/mutationRate)));
    if (random % 250 == 0) {
       // offspring->network->Mutate(600);
        offspring->network->Mutate((int)(network->synapse_count / (50 / (mutationRate*15))));

    }
    else {
      //  offspring->network->Mutate(40);
        offspring->network->Mutate((int)(network->synapse_count / (50 / mutationRate)));
    }
 
    
    
    if (rand() % 2 == 1) {
        offspring->mutationRate = mutationRate * 1.01f;
    }
    else {
        offspring->mutationRate = mutationRate * .99f;
    }
    //if (offspring->mutationRate <= .25f) {
       // offspring->mutationRate = 1.0f;
  //  }
    
    for (auto& neuron : offspring->network->neurons) {
        neuron.activation = 0;
        neuron.refactory = 0;
    }
    offspring->generation = generation + 1;
    offspring->input = input;
    offspring->output = output;
    offspring->health = 5000.0f;
    offspring->air = air;
    offspring->alive = true;
    offspring->color = color;
    // Cast to int before adding, clamp manually
    if (random % 250 == 0) {
        int newR = color.r + (rand() % 220 - 100);
        offspring->color.r = std::max(0, std::min(255, newR));

        int newG = color.g + (rand() % 220 - 100);
        offspring->color.g = std::max(0, std::min(255, newG));

        int newB = color.b + (rand() % 220 - 100);
        offspring->color.b = std::max(0, std::min(255, newB));
    }
    else {
        int newR = color.r + (rand() % 11 - 5);
        offspring->color.r = std::max(0, std::min(255, newR));

        int newG = color.g + (rand() % 11 - 5);
        offspring->color.g = std::max(0, std::min(255, newG));

        int newB = color.b + (rand() % 11 - 5);
        offspring->color.b = std::max(0, std::min(255, newB));
    }

    // Clone modules from the parent, updating their organism and network pointers.
    for (const auto& mod : modules) {
        auto cloned = mod->Clone(offspring.get(), offspring->network.get());
     //   std::cout << "o-input: " << offspring->inputsUsed << '\n';
       // std::cout << "o-output: " << offspring->outputsUsed << '\n';
      //  cloned->SetIndexOffsets(offspring->inputsUsed, offspring->outputsUsed);
      //  std::cout << "input: " << cloned.get()->inputStartingNeuron << '\n';
      //  std::cout << "output: " << cloned.get()->outputStartingNeuron << '\n';

        //why wouldn't output crash it if outputs used is added?
        offspring->modules.push_back(std::move(cloned));
        offspring->inputsUsed += mod.get()->input;
        offspring->outputsUsed += mod.get()->output;
       

    }
    if (network->neurons.size() != offspring->network->neurons.size()) {
        std::cout << "[Reproduce] Parent neurons: " << network->neurons.size()
            << ", Offspring neurons: " << offspring->network->neurons.size() << '\n';
    }

    return offspring;
}
