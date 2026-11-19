#include "brain_io.h"

#include <fstream>
#include <sstream>
#include "world.h"

namespace {
constexpr const char* kMagic = "nworks-organism";
constexpr int kVersion = 1;
}  // namespace

bool SaveOrganism(const Organism& o, const Config& cfg, const std::string& path, const std::string& note) {
    std::ofstream out(path);
    if (!out) return false;
    out << kMagic << ' ' << kVersion << '\n';
    out << "id " << o.id << '\n';
    out << "generation " << o.generation << '\n';
    out.precision(9);
    out << "mutation_rate " << o.mutationRate << '\n';
    out << "color " << int(o.color.r) << ' ' << int(o.color.g) << ' ' << int(o.color.b) << '\n';
    out << "body " << cfg.vision << ' ' << cfg.visionSectors << ' ' << cfg.poison << ' ' << cfg.plasticity << ' '
        << cfg.signals << ' ' << cfg.cooperativeFood << ' ' << cfg.predators << ' ' << cfg.organismSensor << ' '
        << cfg.positionSensor << ' ' << cfg.slices << '\n';
    if (!note.empty()) out << "note " << note << '\n';
    o.brain.Save(out);
    return bool(out);
}

bool LoadOrganism(const std::string& path, Config& cfg, SavedOrganism& out, std::string& error) {
    std::ifstream in(path);
    if (!in) {
        error = "can't open " + path;
        return false;
    }
    std::string magic;
    int version = 0;
    if (!(in >> magic >> version) || magic != kMagic || version != kVersion) {
        error = path + " is not an Nworks organism file";
        return false;
    }
    std::string key;
    while (in >> key) {
        if (key == "id") in >> out.id;
        else if (key == "generation") in >> out.generation;
        else if (key == "mutation_rate") in >> out.mutationRate;
        else if (key == "color") {
            int r = 0, g = 0, b = 0;
            in >> r >> g >> b;
            out.color = Rgb{uint8_t(r), uint8_t(g), uint8_t(b)};
        } else if (key == "body") {
            in >> cfg.vision >> cfg.visionSectors >> cfg.poison >> cfg.plasticity >> cfg.signals >>
                cfg.cooperativeFood >> cfg.predators >> cfg.organismSensor >> cfg.positionSensor >> cfg.slices;
        } else if (key == "note") {
            std::getline(in, out.note);
            if (!out.note.empty() && out.note[0] == ' ') out.note.erase(0, 1);
        } else if (key == "network") {
            // Hand the rest of the file, header included, to the network.
            std::stringstream rest;
            rest << key << in.rdbuf();
            if (!out.brain.Load(rest, error)) {
                error = path + ": " + error;
                return false;
            }
            if (!in && !in.eof()) {
                error = path + ": read error";
                return false;
            }
            return true;
        } else {
            error = path + ": unknown line '" + key + "'";
            return false;
        }
        if (!in) {
            error = path + ": bad value for '" + key + "'";
            return false;
        }
    }
    error = path + ": no network";
    return false;
}

bool MakeOrganism(const SavedOrganism& saved, const World& world, float x, float y, Organism& out,
                  std::string& error) {
    Organism o(saved.id, x, y, world.cfg);
    world.AddBodyModules(o);
    if (o.InputsUsed() != saved.brain.NumInputs() || o.OutputsUsed() != saved.brain.NumOutputs()) {
        error = "brain has " + std::to_string(saved.brain.NumInputs()) + " inputs / " +
                std::to_string(saved.brain.NumOutputs()) + " outputs, body has " + std::to_string(o.InputsUsed()) +
                " / " + std::to_string(o.OutputsUsed());
        return false;
    }
    o.brain = saved.brain;
    o.brain.ResetState();
    o.generation = saved.generation;
    o.mutationRate = saved.mutationRate;
    o.color = saved.color;
    out = std::move(o);
    return true;
}
