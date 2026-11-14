#ifndef RETRO_LLM_ENGINE_H
#define RETRO_LLM_ENGINE_H
#include <stdint.h>

void llm_reset(void);
void llm_step(int tok, int8_t *logits);          /* one token -> 36 int8 logits */
int llm_sample(const int8_t *logits, uint16_t *rng, int temp_idx);
uint16_t llm_rng_next(uint16_t *x);

extern const char LLM_VOCAB[];
#endif
