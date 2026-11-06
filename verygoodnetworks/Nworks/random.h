#pragma once
#include <cstdint>
#include <random>

// One shared random generator, so a run can be repeated from its seed.
namespace rng {

void Seed(uint32_t seed);  // 0 picks a random seed
uint32_t CurrentSeed();
std::mt19937& Engine();

int Int(int lo, int hi);          // uniform in [lo, hi]
float Float(float lo, float hi);  // uniform in [lo, hi)
bool Chance(float probability);
int RoundRandomly(float x);       // floor(x), plus 1 with probability frac(x)

}  // namespace rng
