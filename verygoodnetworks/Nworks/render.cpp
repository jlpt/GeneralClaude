#include "render.h"

#include <algorithm>
#include <cmath>
#include <string>
#include "world.h"

namespace {

std::string WithCommas(long long value) {
    std::string digits = std::to_string(value < 0 ? -value : value);
    for (int i = int(digits.size()) - 3; i > 0; i -= 3) digits.insert(size_t(i), ",");
    return value < 0 ? "-" + digits : digits;
}

struct Rank {
    size_t below;
    const char* name;
};

const char* NeuronRank(size_t neurons) {
    static const Rank ranks[] = {{10, "Spark"},        {25, "Cell"},          {50, "Node"},      {100, "Core"},
                                 {250, "Cluster"},     {500, "Lobe"},         {1000, "Conglomerate"},
                                 {2500, "System"},     {5000, "Cerebrum"},    {10000, "Cortex"},
                                 {25000, "Overmind"},  {50000, "Omnipresence"}, {100000, "Transcendence"}};
    for (const Rank& r : ranks) {
        if (neurons < r.below) return r.name;
    }
    return "Singularity";
}

const char* SynapseRank(size_t synapses) {
    static const Rank ranks[] = {{250, "Sparse"},       {500, "Linked"},       {1000, "Chained"},
                                 {2500, "Integrated"},  {5000, "Webbed"},      {10000, "Latticed"},
                                 {50000, "Connected"},  {100000, "Meshed"},    {250000, "Hyperlinked"},
                                 {1000000, "Quantic"},  {5000000, "Metalinked"}, {10000000, "Entangled"}};
    for (const Rank& r : ranks) {
        if (synapses < r.below) return r.name;
    }
    return "Infinite";
}

Color RankColor(size_t synapses) {
    struct Tier {
        size_t below;
        Color color;
    };
    static const Tier tiers[] = {{250, WHITE},     {500, BLUE},      {1000, GREEN},      {2500, DARKGREEN},
                                 {5000, DARKBLUE}, {10000, PURPLE},  {25000, GOLD},      {50000, ORANGE},
                                 {100000, MAROON}, {250000, LIME},   {500000, SKYBLUE},  {1000000, VIOLET},
                                 {2500000, MAGENTA}, {5000000, YELLOW}, {10000000, RED}};
    for (const Tier& t : tiers) {
        if (synapses < t.below) return t.color;
    }
    return BLACK;
}

std::string RankName(size_t neurons, size_t synapses) {
    return std::string(SynapseRank(synapses)) + "-" + NeuronRank(neurons);
}

unsigned char Byte(float t) { return (unsigned char)(255.0f * std::clamp(t, 0.0f, 1.0f)); }

// 0 = red, 1 = green.
Color RedToGreen(float t) { return Color{Byte(1.0f - t), Byte(t), 0, 255}; }

Color ToColor(Rgb c, unsigned char alpha = 255) { return Color{c.r, c.g, c.b, alpha}; }

void DrawDisc(Vector2 center, float radius, Color fill) {
    DrawCircleV(center, radius + 3.0f, BLACK);
    DrawCircleV(center, radius, fill);
}

}  // namespace

void DrawWorld(const World& world, uint64_t glowId, uint64_t selectedId) {
    const Config& cfg = world.cfg;
    DrawRectangleRec(Rectangle{-4.0f, -4.0f, cfg.worldWidth + 8.0f, cfg.worldHeight + 8.0f}, BLACK);
    DrawRectangleRec(Rectangle{0.0f, 0.0f, cfg.worldWidth, cfg.worldHeight}, SKYBLUE);

    for (const Slice& s : world.slices) {
        const unsigned char alpha = Byte(s.Opacity());
        if (alpha == 0) continue;
        DrawLineEx({s.x1, s.y1}, {s.x2, s.y2}, Slice::kWidth, Color{255, 255, 255, alpha});
        DrawLineEx({s.x1, s.y1}, {s.x2, s.y2}, Slice::kWidth - 1.0f, Color{0, 0, 0, alpha});
    }
    // Season patches: the blooming one is outlined strongly.
    if (cfg.seasons) {
        for (int p = 0; p < World::kPatchCount; ++p) {
            const bool blooming = p == world.Season();
            DrawCircleV({world.PatchX(p), world.PatchY(p)}, cfg.patchRadius, Fade(DARKGREEN, blooming ? 0.25f : 0.08f));
        }
    }

    for (const Resource& r : world.food) DrawDisc({r.x, r.y}, r.radius, RED);
    for (const Resource& r : world.airPockets) DrawDisc({r.x, r.y}, r.radius, BLUE);

    // Springs: the bright centre shrinks as the water runs out.
    for (const Spring& s : world.springs) {
        DrawDisc({s.x, s.y}, s.radius, DARKBLUE);
        const float fill = std::sqrt(std::clamp(s.water / cfg.springCapacity, 0.0f, 1.0f));
        DrawCircleV({s.x, s.y}, s.radius * fill, BLUE);
    }

    const uint64_t focusId = selectedId ? selectedId : glowId;
    for (const Organism& o : world.organisms) {
        const Vector2 center{o.x, o.y};
        if (o.id == glowId) DrawCircleV(center, o.radius + 32.0f, ToColor(o.color, 192));
        if (o.id == selectedId) DrawRing(center, o.radius + 5.0f, o.radius + 8.0f, 0.0f, 360.0f, 32, WHITE);
        if (o.id == focusId && cfg.vision) DrawCircleLines(int(o.x), int(o.y), cfg.visionRange, Fade(WHITE, 0.4f));
        DrawDisc(center, o.radius, ToColor(o.color));
    }
}

void DrawWorldStats(const World& world, int x, int y) {
    DrawText(TextFormat("Difficulty: %.0f%%", world.difficulty * 100.0f), x, y, 18, WHITE);
    if (world.cfg.seasons) {
        DrawText(TextFormat("Season: %d of %d", world.Season() + 1, World::kPatchCount), x, y + 20, 18, WHITE);
    }
    DrawText(TextFormat("Biggest brain ever: %s", WithCommas((long long)world.stats.largestBrainEver).c_str()), x,
             y + 40, 18, WHITE);
}

void DrawBrain(const NeuralNetwork& brain, Rectangle area) {
    const std::vector<Neuron>& neurons = brain.Neurons();
    auto at = [&area](const Neuron& n) {
        return Vector2{area.x + n.layoutX * area.width, area.y + n.layoutY * area.height};
    };

    // Synapses: red (inhibitory) to green (excitatory); yellow while carrying a spike.
    if (brain.SynapseCount() <= 5000) {
        for (const Neuron& n : neurons) {
            const Vector2 from = at(n);
            for (const Connection& c : n.connections) {
                const Vector2 to = at(neurons[c.Target()]);
                DrawLineEx(from, to, 3.0f, BLACK);
                DrawLineV(from, to, n.spiked ? YELLOW : RedToGreen((c.Weight() + 1.0f) * 0.5f));
            }
        }
    }

    // Neurons: inputs grey by sensor value, hidden red-to-green by how close
    // they are to firing, outputs by level; yellow on the step they spike.
    if (neurons.size() <= 1000) {
        const float radius = std::clamp(150.0f / std::sqrt(float(neurons.size())), 3.0f, 10.0f);
        for (size_t i = 0; i < neurons.size(); ++i) {
            const Neuron& n = neurons[i];
            Color fill;
            switch (n.type) {
            case NeuronType::Input: {
                const unsigned char v = Byte(brain.Input(int(i)));
                fill = Color{v, v, v, 255};
                break;
            }
            case NeuronType::Output:
                fill = Color{Byte(n.potential), 255, Byte(1.0f - n.potential), 255};
                break;
            case NeuronType::Hidden:
                fill = RedToGreen(n.potential / n.threshold);
                break;
            }
            if (n.spiked) fill = YELLOW;
            DrawDisc(at(n), radius, fill);
        }
    }
}

void DrawOrganismStats(const Organism& o, int x, int y) {
    const size_t neurons = o.brain.NeuronCount();
    const size_t synapses = o.brain.SynapseCount();
    DrawText(RankName(neurons, synapses).c_str(), x, y, 20, RankColor(synapses));
    DrawText(TextFormat("Synapses: %s", WithCommas((long long)synapses).c_str()), x, y + 16, 20, WHITE);
    DrawText(TextFormat("Neurons: %s", WithCommas((long long)neurons).c_str()), x, y + 32, 20, WHITE);
    DrawText(TextFormat("Gen: %s", WithCommas(o.generation).c_str()), x, y + 48, 20, WHITE);
    DrawText(TextFormat("Mutation Rate: %.2fx", o.mutationRate), x, y + 64, 20, WHITE);
    DrawText(TextFormat("Health: %s", WithCommas((long long)o.health).c_str()), x, y + 80, 20, WHITE);
    DrawText(TextFormat("Air: %s", WithCommas((long long)o.air).c_str()), x, y + 96, 20, WHITE);
}

void DrawPopulationStats(const World& world, int x, int y) {
    const size_t neurons = world.TotalNeurons();
    const size_t synapses = world.TotalSynapses();
    DrawText(TextFormat("Total Organisms: %s", WithCommas((long long)world.organisms.size()).c_str()), x, y, 20,
             WHITE);
    DrawText(TextFormat("Total Neurons: %s", WithCommas((long long)neurons).c_str()), x + 240, y, 20, WHITE);
    DrawText(TextFormat("Total Synapses: %s", WithCommas((long long)synapses).c_str()), x + 500, y, 20, WHITE);
    DrawText(RankName(neurons, synapses).c_str(), x + 760, y, 20, RankColor(synapses));
    DrawText(TextFormat("Step: %s", WithCommas((long long)world.stats.steps).c_str()), x + 1060, y, 20, WHITE);
}

void DrawTooltip(const Organism& o, Vector2 screenPos) {
    const int x = int(screenPos.x) + 16;
    const int y = int(screenPos.y) + 16;
    const size_t neurons = o.brain.NeuronCount();
    const size_t synapses = o.brain.SynapseCount();
    DrawRectangle(x - 6, y - 4, 260, 88, Fade(BLACK, 0.5f));
    DrawText(RankName(neurons, synapses).c_str(), x, y, 20, RankColor(synapses));
    DrawText(TextFormat("Synapses: %s", WithCommas((long long)synapses).c_str()), x, y + 16, 20, WHITE);
    DrawText(TextFormat("Neurons: %s", WithCommas((long long)neurons).c_str()), x, y + 32, 20, WHITE);
    DrawText(TextFormat("Health: %s", WithCommas((long long)o.health).c_str()), x, y + 48, 20, WHITE);
    DrawText(TextFormat("Gen: %s", WithCommas(o.generation).c_str()), x, y + 64, 20, WHITE);
}

const Organism* OrganismAt(const World& world, Vector2 worldPos) {
    // Search backwards so the organism drawn on top wins.
    for (auto it = world.organisms.rbegin(); it != world.organisms.rend(); ++it) {
        if (CheckCollisionPointCircle(worldPos, Vector2{it->x, it->y}, it->radius)) return &*it;
    }
    return nullptr;
}
