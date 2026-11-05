#include "random.h"

namespace rng {
namespace {

std::mt19937 engine{std::random_device{}()};
uint32_t seedInUse = 0;

}  // namespace

void Seed(uint32_t seed) {
    std::random_device device;
    while (seed == 0) seed = device();
    seedInUse = seed;
    engine.seed(seed);
}

uint32_t CurrentSeed() { return seedInUse; }

std::mt19937& Engine() { return engine; }

int Int(int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(engine); }

float Float(float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(engine); }

bool Chance(float probability) { return Float(0.0f, 1.0f) < probability; }

}  // namespace rng
