#pragma once
#include <cstdint>
#include <string>
#include "config.h"
#include "neural_network.h"
#include "organism.h"

class World;

// A saved organism: its brain plus the inherited traits, and the config
// switches that decide which inputs and outputs the brain is wired to.
//
// File format (text, one item per line):
//   nworks-organism 1
//   id <id>
//   generation <n>
//   mutation_rate <x>
//   color <r> <g> <b>
//   body <vision> <visionSectors> <poison> <plasticity> <signals> <cooperativeFood>
//        <predators> <organismSensor> <positionSensor> <slices>
//   note <free text>            (optional, for example how it scored)
//   network ...                 (see NeuralNetwork::Save)
struct SavedOrganism {
    uint64_t id = 0;
    int generation = 0;
    float mutationRate = 1.0f;
    Rgb color;
    std::string note;
    NeuralNetwork brain;
};

bool SaveOrganism(const Organism& o, const Config& cfg, const std::string& path, const std::string& note = "");

// Reads a saved organism and switches the body settings in `cfg` to match it.
bool LoadOrganism(const std::string& path, Config& cfg, SavedOrganism& out, std::string& error);

// Builds a live organism at (x, y) from a saved one. `world.cfg` must have the
// saved body settings (from LoadOrganism). Returns false if the brain's input
// or output count doesn't match the body.
bool MakeOrganism(const SavedOrganism& saved, const World& world, float x, float y, Organism& out,
                  std::string& error);
