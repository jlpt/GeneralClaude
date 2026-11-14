/* Host/qemu test: print N generated token ids for a seed and temperature. */
#include <stdio.h>
#include <stdlib.h>
#include "engine.h"

int main(int argc, char **argv)
{
    int n = argc > 1 ? atoi(argv[1]) : 200;
    uint16_t rng = argc > 2 ? (uint16_t)strtol(argv[2], 0, 0) : 0x1234;
    int temp = argc > 3 ? atoi(argv[3]) : 2;
    int8_t logits[64];
    int tok = 0;
    if (!rng)
        rng = 1;
    llm_reset();
    for (int i = 0; i < n; i++) {
        llm_step(tok, logits);
        tok = llm_sample(logits, &rng, temp);
        printf("%d ", tok);
    }
    printf("\n");
    return 0;
}
