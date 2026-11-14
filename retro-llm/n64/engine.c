/*
 * retro-llm inference engine in portable C (integer only, bit-exact with
 * train/ref.py). Used by the N64 build and by the host-side test.
 */
#include "engine.h"
#include "gen/model.h"

static int8_t KC[LLM_L][LLM_T][LLM_D];
static int8_t VC[LLM_L][LLM_T][LLM_D];
static int t_pos;

static int32_t rr(int32_t a, int s)
{
    return s ? (a + (1 << (s - 1))) >> s : a;
}

static int8_t clamp8(int32_t v)
{
    return v > 127 ? 127 : v < -127 ? -127 : (int8_t)v;
}

/* y[j] = clamp8(rr(sum_i x[i] * W[i][j], s)) */
static void matvec(const int8_t *x, int nin, const int8_t *w, int nout, int s, int8_t *y)
{
    int32_t acc[LLM_F];
    for (int j = 0; j < nout; j++)
        acc[j] = 0;
    for (int i = 0; i < nin; i++) {
        int32_t a = x[i];
        if (!a)
            continue;
        const int8_t *row = w + i * nout;
        for (int j = 0; j < nout; j++)
            acc[j] += a * row[j];
    }
    for (int j = 0; j < nout; j++)
        y[j] = clamp8(rr(acc[j], s));
}

void llm_reset(void)
{
    t_pos = 0;
}

void llm_step(int tok, int8_t *logits)
{
    int8_t x[LLM_D], q[LLM_D], o[LLM_D], tv[LLM_F], h[LLM_F];
    int slot = t_pos % LLM_T;
    int n = t_pos + 1 < LLM_T ? t_pos + 1 : LLM_T;

    for (int i = 0; i < LLM_D; i++)
        x[i] = LLM_EMB[tok * LLM_D + i];

    for (int l = 0; l < LLM_L; l++) {
        const int8_t *W = LLM_LAYERS[l];
        const uint8_t *sh = LLM_SHIFTS + l * 6;
        matvec(x, LLM_D, W + 0x0000, LLM_D, sh[0], q);
        matvec(x, LLM_D, W + 0x0400, LLM_D, sh[1], KC[l][slot]);
        matvec(x, LLM_D, W + 0x0800, LLM_D, sh[2], VC[l][slot]);

        for (int hd = 0; hd < LLM_H; hd++) {
            int c0 = hd * LLM_HD;
            int32_t sc[LLM_T], m = -0x7FFFFFFF;
            uint32_t e[LLM_T], esum = 0;
            for (int j = 0; j < n; j++) {
                int32_t dot = 0;
                for (int c = 0; c < LLM_HD; c++)
                    dot += q[c0 + c] * KC[l][j][c0 + c];
                int dist = (slot - j) & (LLM_T - 1);
                sc[j] = rr(dot, LLM_SS) - LLM_ALIBI[hd * LLM_T + dist];
                if (sc[j] > m)
                    m = sc[j];
            }
            for (int j = 0; j < n; j++) {
                int32_t d = m - sc[j];
                e[j] = LLM_EXP_ATT[d > 255 ? 255 : d];
                esum += e[j];
            }
            int32_t acc[LLM_HD] = {0};
            for (int j = 0; j < n; j++) {
                uint32_t p = (e[j] << 7) / esum;
                if (p > 127)
                    p = 127;
                for (int c = 0; c < LLM_HD; c++)
                    acc[c] += (int32_t)p * VC[l][j][c0 + c];
            }
            for (int c = 0; c < LLM_HD; c++)
                o[c0 + c] = clamp8(rr(acc[c], 7));
        }

        matvec(o, LLM_D, W + 0x0C00, LLM_D, sh[3], tv);
        for (int i = 0; i < LLM_D; i++)
            x[i] = clamp8(x[i] + tv[i]);
        matvec(x, LLM_D, W + 0x1000, LLM_F, sh[4], h);
        for (int i = 0; i < LLM_F; i++)
            if (h[i] < 0)
                h[i] = 0;
        matvec(h, LLM_F, W + 0x1800, LLM_D, sh[5], tv);
        for (int i = 0; i < LLM_D; i++)
            x[i] = clamp8(x[i] + tv[i]);
    }
    matvec(x, LLM_D, LLM_WOUT, LLM_V, LLM_SHIFT_OUT, logits);
    t_pos++;
}

uint16_t llm_rng_next(uint16_t *x)
{
    uint16_t v = *x;
    v ^= (uint16_t)(v << 7);
    v ^= v >> 9;
    v ^= (uint16_t)(v << 8);
    return *x = v;
}

int llm_sample(const int8_t *logits, uint16_t *rng, int temp_idx)
{
    const uint8_t *tab = LLM_EXPS + temp_idx * 256;
    int lmax = logits[0];
    for (int v = 1; v < LLM_V; v++)
        if (logits[v] > lmax)
            lmax = logits[v];
    uint32_t e[LLM_V], esum = 0;
    for (int v = 0; v < LLM_V; v++)
        esum += e[v] = tab[lmax - logits[v]];
    uint32_t mask = 0;
    for (uint32_t t = esum; t; t >>= 1)
        mask = (mask << 1) | 1;
    uint32_t r;
    do
        r = llm_rng_next(rng) & mask;
    while (r >= esum);
    uint32_t acc = 0;
    for (int v = 0; v < LLM_V; v++) {
        acc += e[v];
        if (r < acc)
            return v;
    }
    return LLM_V - 1;
}
