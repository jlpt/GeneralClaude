// C interface to Nworks for brains trained outside it (experiments/train_rl.py
// loads this as a shared library with ctypes).
//
// Two kinds of batch, both with externally controlled creatures (see
// ApplyAction for the 18 discrete actions):
//   nwb_*  training episodes: one creature alone in a full world (food,
//          poison, springs, seasons, predators at a fixed difficulty),
//          restarted automatically when it dies or the episode ends.
//   nws_*  test battery: the same Scenario code nworks_iq uses.
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include "common.h"
#include "scenarios.h"

namespace {

struct Episode {
    std::unique_ptr<World> world;
    int t = 0;
};

struct Batch {
    Config cfg;
    float difficulty = 0.3f;
    bool randomPoison = false;  // the poisonous colour is picked at random for each episode
    int episodeSteps = 3000;
    uint32_t nextSeed = 1;
    std::unique_ptr<Organism> body;  // template, copied into every episode
    std::vector<Episode> envs;
};

void Observe(World& w, float* obs, int dim) {
    if (w.organisms.empty()) {
        std::memset(obs, 0, sizeof(float) * size_t(dim));
        return;
    }
    Organism& o = w.organisms[0];
    for (auto& m : o.modules) m->Sense(o, w);
    for (int i = 0; i < dim; ++i) obs[i] = o.brain.Input(i);
}

void ResetEpisode(Batch& b, Episode& e) {
    Config c = b.cfg;
    c.seed = b.nextSeed++;
    e.world = std::make_unique<World>(c);
    e.world->difficulty = b.difficulty;
    e.t = 0;
    std::mt19937 g(c.seed * 2654435761u + 5u);
    if (b.randomPoison) {
        e.world->poisonKindOverride = int(g() % 2);
        for (Resource& r : e.world->food) r.kind = uint8_t(g() % 2);  // initial food was coloured before the pick
    }
    Organism o = CloneOf(*b.body, 1, c);
    o.external = true;
    o.x = std::uniform_real_distribution<float>(40.0f, c.worldWidth - 40.0f)(g);
    o.y = std::uniform_real_distribution<float>(40.0f, c.worldHeight - 40.0f)(g);
    o.health = c.maxHealth;
    o.air = c.maxAir;
    e.world->organisms.push_back(std::move(o));
}

struct ScenarioBatch {
    std::vector<Scenario> trials;
    int dim = 0;
};

}  // namespace

extern "C" {

int nw_num_actions() { return kNumActions; }
int nw_num_tests() { return int(Test::Count); }
const char* nw_test_name(int t) { return TestName(Test(t)); }

// --- training episodes -------------------------------------------------------
// overrides: space-separated Config "key=value" settings, plus "difficulty=D" and
// "random_poison=1" (the poisonous colour is random for each episode).
void* nwb_create(int n, uint32_t seed, const char* overrides, int episodeSteps) {
    auto b = new Batch;
    std::string text = overrides ? overrides : "", rest;
    for (size_t pos = 0; pos < text.size();) {  // pull out difficulty=, pass the rest to ApplyOverrides
        size_t end = text.find(' ', pos);
        if (end == std::string::npos) end = text.size();
        const std::string item = text.substr(pos, end - pos);
        if (item.rfind("difficulty=", 0) == 0) b->difficulty = std::stof(item.substr(11));
        else if (item == "random_poison=1") b->randomPoison = true;
        else rest += " " + item;
        pos = end + 1;
    }
    if (!ApplyOverrides(b->cfg, rest)) { delete b; return nullptr; }
    b->cfg.minPopulation = 0;
    b->cfg.foodToReproduce = 1000000000;
    b->cfg.curriculumWindow = 0;
    b->cfg.foodCurriculum = false;
    b->episodeSteps = episodeSteps;
    b->nextSeed = seed * 1000003u + 1u;
    b->body = std::make_unique<Organism>(MakeExternalBody(b->cfg));
    b->envs.resize(size_t(n));
    for (Episode& e : b->envs) ResetEpisode(*b, e);
    return b;
}

int nwb_obs_dim(void* p) { return static_cast<Batch*>(p)->body->brain.NumInputs(); }

void nwb_observe(void* p, float* obs) {
    auto& b = *static_cast<Batch*>(p);
    const int dim = b.body->brain.NumInputs();
    for (size_t i = 0; i < b.envs.size(); ++i) Observe(*b.envs[i].world, obs + i * size_t(dim), dim);
}

// One step for every episode. stats gets 3 ints per episode: good bites,
// poison bites, died (this step). Finished episodes restart; obs is the new
// observation (of the fresh episode where one restarted).
void nwb_step(void* p, const int32_t* actions, float* obs, float* rewards, uint8_t* dones, int32_t* stats) {
    auto& b = *static_cast<Batch*>(p);
    const int dim = b.body->brain.NumInputs();
    for (size_t i = 0; i < b.envs.size(); ++i) {
        Episode& e = b.envs[i];
        World& w = *e.world;
        if (!w.organisms.empty()) ApplyAction(w.organisms[0], actions[i]);
        const uint64_t good0 = w.stats.goodBites, poison0 = w.stats.poisonBites;
        w.Step();
        ++e.t;
        const int good = int(w.stats.goodBites - good0), poison = int(w.stats.poisonBites - poison0);
        const bool died = w.organisms.empty();
        rewards[i] = float(good) - float(poison) - (died ? 2.0f : 0.0f);
        stats[3 * i + 0] = good;
        stats[3 * i + 1] = poison;
        stats[3 * i + 2] = died ? 1 : 0;
        dones[i] = (died || e.t >= b.episodeSteps) ? 1 : 0;
        if (dones[i]) ResetEpisode(b, e);
        Observe(*e.world, obs + i * size_t(dim), dim);
    }
}

void nwb_destroy(void* p) { delete static_cast<Batch*>(p); }

// --- test battery ----------------------------------------------------------------
// n trials of one test; trial k uses the same seed nworks_iq gives its k-th trial.
void* nws_create(int test, int n, const char* overrides) {
    Config cfg;
    if (!ApplyOverrides(cfg, overrides ? overrides : "")) return nullptr;
    auto s = new ScenarioBatch;
    Organism body = MakeExternalBody(cfg);
    s->dim = body.brain.NumInputs();
    s->trials.reserve(size_t(n));
    for (int k = 0; k < n; ++k)
        s->trials.emplace_back(Test(test), cfg, uint32_t((test + 1) * 100000 + k), CloneOf(body, 1, cfg));
    for (Scenario& sc : s->trials) {
        if (Organism* o = sc.Creature()) o->external = true;
        sc.Sense();
    }
    return s;
}

int nws_obs_dim(void* p) { return static_cast<ScenarioBatch*>(p)->dim; }

void nws_observe(void* p, float* obs) {
    auto& s = *static_cast<ScenarioBatch*>(p);
    for (size_t i = 0; i < s.trials.size(); ++i) {
        Organism* o = s.trials[i].Creature();
        for (int j = 0; j < s.dim; ++j) obs[i * size_t(s.dim) + size_t(j)] = o ? o->brain.Input(j) : 0.0f;
    }
}

// Steps every unfinished trial with its action. Returns how many are still running.
int nws_step(void* p, const int32_t* actions, uint8_t* dones) {
    auto& s = *static_cast<ScenarioBatch*>(p);
    int running = 0;
    for (size_t i = 0; i < s.trials.size(); ++i) {
        Scenario& sc = s.trials[i];
        if (!sc.Done()) {
            if (Organism* o = sc.Creature()) ApplyAction(*o, actions[i]);
            sc.Step();
            sc.Sense();
        }
        dones[i] = sc.Done() ? 1 : 0;
        running += sc.Done() ? 0 : 1;
    }
    return running;
}

void nws_results(void* p, uint8_t* success, float* score, uint8_t* bit) {
    auto& s = *static_cast<ScenarioBatch*>(p);
    for (size_t i = 0; i < s.trials.size(); ++i) {
        success[i] = s.trials[i].Success() ? 1 : 0;
        score[i] = s.trials[i].Score();
        bit[i] = s.trials[i].Bit() ? 1 : 0;
    }
}

// PoisonLearning details, 4 ints per trial: ate in round 1, round-1 pick safe,
// bites in rounds 2-6, safe bites in rounds 2-6.
void nws_learning_results(void* p, int32_t* out) {
    auto& s = *static_cast<ScenarioBatch*>(p);
    for (size_t i = 0; i < s.trials.size(); ++i) {
        const Scenario& sc = s.trials[i];
        out[4 * i + 0] = sc.FirstAte() ? 1 : 0;
        out[4 * i + 1] = sc.FirstSafe() ? 1 : 0;
        out[4 * i + 2] = sc.LaterBites();
        out[4 * i + 3] = sc.LaterSafe();
    }
}

void nws_destroy(void* p) { delete static_cast<ScenarioBatch*>(p); }

}  // extern "C"
