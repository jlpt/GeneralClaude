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

// Many creatures per world, each one an agent slot. A creature that dies is
// replaced by a fresh one in its slot; the whole world restarts every
// episodeSteps steps.
struct MultiEnv {
    std::unique_ptr<World> world;
    std::vector<uint64_t> ids;  // creature id in each slot
    int t = 0;
};

struct MultiBatch {
    Config cfg;
    float difficulty = 0.3f, share = 0.0f;
    int perWorld = 8, episodeSteps = 3000;
    uint32_t nextSeed = 1;
    uint64_t nextId = 1;
    std::unique_ptr<Organism> body;
    std::vector<MultiEnv> envs;
};

Organism* FindCreature(World& w, uint64_t id) {
    for (Organism& o : w.organisms)
        if (o.id == id) return &o;
    return nullptr;
}

void SpawnInSlot(MultiBatch& b, MultiEnv& e, size_t slot, std::mt19937& g) {
    Organism o = CloneOf(*b.body, b.nextId++, b.cfg);
    o.id = b.nextId - 1;
    o.external = true;
    o.x = std::uniform_real_distribution<float>(40.0f, b.cfg.worldWidth - 40.0f)(g);
    o.y = std::uniform_real_distribution<float>(40.0f, b.cfg.worldHeight - 40.0f)(g);
    o.health = b.cfg.maxHealth;
    o.air = b.cfg.maxAir;
    e.ids[slot] = o.id;
    e.world->organisms.push_back(std::move(o));
}

void ResetMulti(MultiBatch& b, MultiEnv& e) {
    Config c = b.cfg;
    c.seed = b.nextSeed++;
    e.world = std::make_unique<World>(c);
    e.world->difficulty = b.difficulty;
    e.t = 0;
    e.ids.assign(size_t(b.perWorld), 0);
    std::mt19937 g(c.seed * 2654435761u + 11u);
    for (size_t s = 0; s < e.ids.size(); ++s) SpawnInSlot(b, e, s, g);
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

// Changes Config settings during training (for curricula). They apply to the
// running worlds and to every episode started afterwards. Returns 0 on an
// unknown setting.
int nwb_set(void* p, const char* overrides) {
    auto& b = *static_cast<Batch*>(p);
    if (!ApplyOverrides(b.cfg, overrides ? overrides : "")) return 0;
    for (Episode& e : b.envs) ApplyOverrides(e.world->cfg, overrides);
    return 1;
}

// Everything in episode i, for replays: 4 floats per thing (kind, x, y, extra).
// kind 0 creature (extra = health 0..1), 1 good food / 2 poison (extra = 1 if
// visible), 3 spring (extra = water 0..1), 4 predator (extra = 1 if resting),
// 5 big food. Returns how many things were written (at most cap).
int nwb_state(void* p, int i, float* out, int cap) {
    auto& b = *static_cast<Batch*>(p);
    const World& w = *b.envs[size_t(i)].world;
    int n = 0;
    auto put = [&](float kind, float x, float y, float extra) {
        if (n >= cap) return;
        out[4 * n + 0] = kind;
        out[4 * n + 1] = x;
        out[4 * n + 2] = y;
        out[4 * n + 3] = extra;
        ++n;
    };
    for (const Resource& r : w.food) put(r.kind == w.PoisonKind() ? 2.0f : 1.0f, r.x, r.y, w.FoodVisible(r) ? 1.0f : 0.0f);
    for (const Resource& r : w.bigFood) put(5.0f, r.x, r.y, 1.0f);
    for (const Spring& s : w.springs) put(3.0f, s.x, s.y, s.water / w.cfg.springCapacity);
    for (int k = 0; k < w.ActivePredators(); ++k) {
        const Predator& pr = w.predators[size_t(k)];
        put(4.0f, pr.x, pr.y, pr.rest > 0 ? 1.0f : 0.0f);
    }
    for (const Organism& o : w.organisms) put(0.0f, o.x, o.y, o.health / w.cfg.maxHealth);
    return n;
}

// --- many creatures per world ----------------------------------------------------
// n worlds x perWorld creatures = n * perWorld agent slots. overrides as for
// nwb_create, plus "share=S": each creature's reward is (1 - S) x its own +
// S x the average of its world (S = 0: selfish, 1: fully shared).
void* nwm_create(int n, int perWorld, uint32_t seed, const char* overrides, int episodeSteps) {
    auto b = new MultiBatch;
    std::string text = overrides ? overrides : "", rest;
    for (size_t pos = 0; pos < text.size();) {
        size_t end = text.find(' ', pos);
        if (end == std::string::npos) end = text.size();
        const std::string item = text.substr(pos, end - pos);
        if (item.rfind("difficulty=", 0) == 0) b->difficulty = std::stof(item.substr(11));
        else if (item.rfind("share=", 0) == 0) b->share = std::stof(item.substr(6));
        else rest += " " + item;
        pos = end + 1;
    }
    if (!ApplyOverrides(b->cfg, rest)) { delete b; return nullptr; }
    b->cfg.minPopulation = 0;
    b->cfg.foodToReproduce = 1000000000;
    b->cfg.curriculumWindow = 0;
    b->cfg.foodCurriculum = false;
    b->perWorld = perWorld;
    b->episodeSteps = episodeSteps;
    b->nextSeed = seed * 1000003u + 1u;
    b->body = std::make_unique<Organism>(MakeExternalBody(b->cfg));
    b->envs.resize(size_t(n));
    for (MultiEnv& e : b->envs) ResetMulti(*b, e);
    return b;
}

int nwm_obs_dim(void* p) { return static_cast<MultiBatch*>(p)->body->brain.NumInputs(); }
int nwm_num_actions() { return kNumActionsWithCall; }

void nwm_observe(void* p, float* obs) {
    auto& b = *static_cast<MultiBatch*>(p);
    const int dim = b.body->brain.NumInputs();
    size_t k = 0;
    for (MultiEnv& e : b.envs) {
        for (uint64_t id : e.ids) {
            float* out = obs + (k++) * size_t(dim);
            Organism* o = FindCreature(*e.world, id);
            if (!o) { std::memset(out, 0, sizeof(float) * size_t(dim)); continue; }
            for (auto& m : o->modules) m->Sense(*o, *e.world);
            for (int i = 0; i < dim; ++i) out[i] = o->brain.Input(i);
        }
    }
}

// Same outputs as nwb_step, one entry per agent slot. stats also gets a 4th
// int per slot: 1 if the creature called this step.
void nwm_step(void* p, const int32_t* actions, float* obs, float* rewards, uint8_t* dones, int32_t* stats) {
    auto& b = *static_cast<MultiBatch*>(p);
    const size_t per = size_t(b.perWorld);
    for (size_t w = 0; w < b.envs.size(); ++w) {
        MultiEnv& e = b.envs[w];
        World& world = *e.world;
        std::vector<int> good0(per), poison0(per);
        for (size_t s = 0; s < per; ++s) {
            Organism* o = FindCreature(world, e.ids[s]);
            if (o) {
                ApplyActionWithCall(*o, actions[w * per + s]);
                good0[s] = o->goodBitesInLife;
                poison0[s] = o->poisonBitesInLife;
            }
        }
        world.Step();
        ++e.t;
        std::mt19937 g(uint32_t(e.t) * 2654435761u + uint32_t(w) * 97u + b.nextSeed);
        const bool episodeOver = e.t >= b.episodeSteps;
        float total = 0.0f;
        for (size_t s = 0; s < per; ++s) {
            const size_t k = w * per + s;
            Organism* o = FindCreature(world, e.ids[s]);
            int good = 0, poison = 0;
            if (o) {
                good = o->goodBitesInLife - good0[s];
                poison = o->poisonBitesInLife - poison0[s];
            }
            const bool died = o == nullptr;
            rewards[k] = float(good) - float(poison) - (died ? 2.0f : 0.0f);
            total += rewards[k];
            stats[4 * k + 0] = good;
            stats[4 * k + 1] = poison;
            stats[4 * k + 2] = died ? 1 : 0;
            stats[4 * k + 3] = (o && o->signal > 0.5f) ? 1 : 0;
            dones[k] = (died || episodeOver) ? 1 : 0;
            if (died && !episodeOver) SpawnInSlot(b, e, s, g);  // a newcomer takes the slot
        }
        if (b.share > 0.0f) {
            const float mean = total / float(per);
            for (size_t s = 0; s < per; ++s) {
                const size_t k = w * per + s;
                rewards[k] = (1.0f - b.share) * rewards[k] + b.share * mean;
            }
        }
        if (episodeOver) ResetMulti(b, e);
    }
    nwm_observe(p, obs);
}

void nwm_destroy(void* p) { delete static_cast<MultiBatch*>(p); }

// --- memory gym -----------------------------------------------------------------
// Trains the core memory skill directly: each episode is a memory scenario
// (food 150 or 250 px away, visible for the first `visible` steps), with fresh
// layouts that never repeat the battery's. Reward +1 for eating the food.
struct Gym {
    Config cfg;
    int visible = 600;
    uint32_t nextSeed = 10000000;
    std::unique_ptr<Organism> body;
    std::vector<std::unique_ptr<Scenario>> envs;
};

void ResetGym(Gym& g, std::unique_ptr<Scenario>& s) {
    const uint32_t seed = g.nextSeed++;
    const Test test = (seed % 2) ? Test::MemoryNear : Test::MemoryFar;
    s = std::make_unique<Scenario>(test, g.cfg, seed, CloneOf(*g.body, 1, g.cfg), g.visible);
    if (Organism* o = s->Creature()) o->external = true;
    s->Sense();
}

void* nwg_create(int n, uint32_t seed, const char* overrides, int visible) {
    auto g = new Gym;
    if (!ApplyOverrides(g->cfg, overrides ? overrides : "")) { delete g; return nullptr; }
    g->visible = visible;
    g->nextSeed = 10000000u + seed * 1000003u;
    g->body = std::make_unique<Organism>(MakeExternalBody(g->cfg));
    g->envs.resize(size_t(n));
    for (auto& s : g->envs) ResetGym(*g, s);
    return g;
}

void nwg_set_visible(void* p, int visible) { static_cast<Gym*>(p)->visible = visible; }
int nwg_obs_dim(void* p) { return static_cast<Gym*>(p)->body->brain.NumInputs(); }

void nwg_observe(void* p, float* obs) {
    auto& g = *static_cast<Gym*>(p);
    const int dim = g.body->brain.NumInputs();
    for (size_t i = 0; i < g.envs.size(); ++i) {
        Organism* o = g.envs[i]->Creature();
        for (int j = 0; j < dim; ++j) obs[i * size_t(dim) + size_t(j)] = o ? o->brain.Input(j) : 0.0f;
    }
}

// Same outputs as nwb_step; stats: success, 0, 0 per episode.
void nwg_step(void* p, const int32_t* actions, float* obs, float* rewards, uint8_t* dones, int32_t* stats) {
    auto& g = *static_cast<Gym*>(p);
    for (size_t i = 0; i < g.envs.size(); ++i) {
        Scenario& s = *g.envs[i];
        if (Organism* o = s.Creature()) ApplyAction(*o, actions[i]);
        s.Step();
        s.Sense();
        rewards[i] = (s.Done() && s.Success()) ? 1.0f : 0.0f;
        stats[3 * i + 0] = (s.Done() && s.Success()) ? 1 : 0;
        stats[3 * i + 1] = 0;
        stats[3 * i + 2] = 0;
        dones[i] = s.Done() ? 1 : 0;
        if (s.Done()) ResetGym(g, g.envs[i]);
    }
    nwg_observe(p, obs);
}

void nwg_destroy(void* p) { delete static_cast<Gym*>(p); }

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
