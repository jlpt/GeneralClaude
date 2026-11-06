#pragma once
#include <cstdint>
#include "raylib.h"

class NeuralNetwork;
class Organism;
class World;

// Everything that draws with raylib lives here; the simulation itself has no
// graphics dependency.

// In world (camera) space. `glowId` gets a halo; `selectedId` gets a thick outline.
void DrawWorld(const World& world, uint64_t glowId, uint64_t selectedId);

// In screen space.
void DrawBrain(const NeuralNetwork& brain, Rectangle area);
void DrawOrganismStats(const Organism& o, int x, int y);
void DrawPopulationStats(const World& world, int x, int y);
void DrawWorldStats(const World& world, int x, int y);  // difficulty, season, biggest brain
void DrawTooltip(const Organism& o, Vector2 screenPos);

const Organism* OrganismAt(const World& world, Vector2 worldPos);
