/*
 * retro-llm for the N64: bare-metal front end (no SDK). Boots through
 * libdragon's open-source IPL3 (compat build), draws into a 320x240 16-bit
 * framebuffer with the VI, reads controller 1 through the PIF.
 */
#include <stdint.h>
#include "engine.h"
#include "gen/model.h"
#include "gen/font.h"

#define REG(a) (*(volatile uint32_t *)(a))
#define VI_BASE 0xA4400000
#define SI_DRAM_ADDR REG(0xA4800000)
#define SI_PIF_RD64 REG(0xA4800004)
#define SI_PIF_WR64 REG(0xA4800010)
#define SI_STATUS REG(0xA4800018)

#define FB_PHYS 0x00100000
#define PIF_BUF_PHYS 0x00300000
static volatile uint16_t *const fb = (volatile uint16_t *)(0xA0000000 | FB_PHYS);
static volatile uint32_t *const pifbuf = (volatile uint32_t *)(0xA0000000 | PIF_BUF_PHYS);

#define W 320
#define H 240
#define COLS 40
#define TXT_LEFT 2
#define TXT_COLS 36
#define TXT_TOP 3
#define TXT_ROWS 24
#define STATUS_ROW 28

#define RGBA(r, g, b) (uint16_t)(((r) << 11) | ((g) << 6) | ((b) << 1) | 1)
#define C_BG RGBA(2, 2, 8)
#define C_TEXT RGBA(31, 31, 31)
#define C_HI RGBA(31, 28, 8)
#define C_DIM RGBA(12, 14, 20)

#define BTN_A 0x8000
#define BTN_START 0x1000
#define BTN_UP 0x0800
#define BTN_DOWN 0x0400

#define COUNT_HZ 46875000u   /* COP0 Count runs at half the 93.75 MHz CPU clock */

void *memset(void *d, int c, unsigned n)
{
    uint8_t *p = d;
    while (n--)
        *p++ = (uint8_t)c;
    return d;
}

void *memcpy(void *d, const void *s, unsigned n)
{
    uint8_t *p = d;
    const uint8_t *q = s;
    while (n--)
        *p++ = *q++;
    return d;
}

static uint32_t count(void)
{
    uint32_t c;
    __asm__ volatile("mfc0 %0, $9" : "=r"(c));
    return c;
}

static void wait_ticks(uint32_t ticks)
{
    uint32_t t0 = count();
    while (count() - t0 < ticks)
        ;
}

static void vi_init(void)
{
    static const uint32_t regs[] = {
        0x00003202,   /* status: 16-bit, resample AA, pixel advance 3 */
        FB_PHYS,      /* origin */
        W,            /* width */
        0x200,        /* v_intr */
        0,            /* v_current */
        0x03E52239,   /* burst */
        0x0000020D,   /* v_sync */
        0x00000C15,   /* h_sync */
        0x0C150C15,   /* h_sync_leap */
        0x006C02EC,   /* h_video */
        0x002501FF,   /* v_video */
        0x000E0204,   /* v_burst */
        0x00000200,   /* x_scale: 320 -> 640 */
        0x00000400,   /* y_scale: 240 lines */
    };
    for (int i = 0; i < 14; i++)
        REG(VI_BASE + i * 4) = regs[i];
}

static void fill(int y0, int y1, uint16_t c)
{
    for (int i = y0 * W; i < y1 * W; i++)
        fb[i] = c;
}

static void draw_char(int col, int row, char ch, uint16_t fg)
{
    int code = (unsigned char)ch;
    if (code < 0x20 || code >= 0x60)
        code = 0x20;
    const uint8_t *g = FONT + (code - 0x20) * 8;
    volatile uint16_t *p = fb + row * 8 * W + col * 8;
    for (int y = 0; y < 8; y++, p += W)
        for (int x = 0; x < 8; x++)
            p[x] = (g[y] & (0x80 >> x)) ? fg : C_BG;
}

static void draw_str(int col, int row, const char *s, uint16_t fg)
{
    while (*s)
        draw_char(col++, row, *s++, fg);
}

static void draw_num(int col, int row, uint32_t v, int width, uint16_t fg)
{
    char buf[12];
    int i = 11;
    buf[i] = 0;
    do {
        buf[--i] = (char)('0' + v % 10);
        v /= 10;
    } while (v && i > 0);
    while (11 - i < width)
        buf[--i] = ' ';
    draw_str(col, row, buf + i, fg);
}

/* ---- controller 1 through the PIF ------------------------------------- */
static void si_wait(void)
{
    while (SI_STATUS & 3)
        ;
}

static uint16_t read_pad(void)
{
    for (int i = 0; i < 16; i++)
        pifbuf[i] = 0;
    pifbuf[0] = 0xFF010401;   /* channel 0: send 1 byte, receive 4, command 1 */
    pifbuf[1] = 0xFFFFFFFF;
    pifbuf[2] = 0xFE000000;   /* end of commands */
    pifbuf[15] = 0x00000001;  /* run joybus */
    si_wait();
    SI_DRAM_ADDR = PIF_BUF_PHYS;
    SI_PIF_WR64 = 0x1FC007C0;
    si_wait();
    SI_DRAM_ADDR = PIF_BUF_PHYS;
    SI_PIF_RD64 = 0x1FC007C0;
    si_wait();
    if (pifbuf[0] & 0x0000C000)   /* no controller / error bits in rx byte */
        return 0;
    return (uint16_t)(pifbuf[1] >> 16);
}

/* ---- IS-Viewer debug log (emulators print it; ignored on real carts) ---- */
#define PI_STATUS REG(0xA4600010)
#define ISV_BASE 0xB3FF0000
static int isv_ok;

static void pi_wait(void)
{
    while (PI_STATUS & 3)
        ;
}

static void isv_detect(void)
{
    pi_wait();
    REG(ISV_BASE) = 0x49533634;   /* "IS64" */
    pi_wait();
    isv_ok = REG(ISV_BASE) == 0x49533634;
}

static void isv_write(const char *s, int n)
{
    if (!isv_ok || n <= 0)
        return;
    for (int i = 0; i < n; i += 4) {
        uint32_t w = 0;
        for (int k = 0; k < 4; k++)
            w = (w << 8) | (uint8_t)(i + k < n ? s[i + k] : 0);
        pi_wait();
        REG(ISV_BASE + 0x20 + i) = w;
    }
    pi_wait();
    REG(ISV_BASE + 0x14) = (uint32_t)n;
}

static void isv_str(const char *s)
{
    int n = 0;
    while (s[n])
        n++;
    isv_write(s, n);
}

static void isv_hex(uint32_t v, int digits)
{
    char b[8];
    for (int i = 0; i < digits; i++)
        b[i] = "0123456789ABCDEF"[(v >> (4 * (digits - 1 - i))) & 15];
    isv_write(b, digits);
}

/* ---- text terminal ----------------------------------------------------- */
static int cur_row, cur_col;

static void scroll_text(void)
{
    volatile uint16_t *dst = fb + TXT_TOP * 8 * W;
    volatile uint16_t *src = dst + 8 * W;
    for (int i = 0; i < (TXT_ROWS - 1) * 8 * W; i++)
        dst[i] = src[i];
    fill((TXT_TOP + TXT_ROWS - 1) * 8, (TXT_TOP + TXT_ROWS) * 8, C_BG);
}

static void newline(void)
{
    cur_col = 0;
    if (cur_row == TXT_ROWS - 1)
        scroll_text();
    else
        cur_row++;
}

static void putc_tok(int tok)
{
    if (tok == 0) {
        newline();
        return;
    }
    if (cur_col >= TXT_COLS) {
        newline();
        if (tok == 1)
            return;
    }
    draw_char(TXT_LEFT + cur_col, TXT_TOP + cur_row, LLM_VOCAB[tok], C_TEXT);
    cur_col++;
}

static const char *const TEMP_TXT[4] = {"0.50", "0.70", "0.85", "1.00"};

static void draw_status(int temp, uint32_t tps, uint32_t ntok)
{
    draw_str(1, STATUS_ROW, "TEMP", C_DIM);
    draw_str(6, STATUS_ROW, TEMP_TXT[temp], C_HI);
    draw_num(12, STATUS_ROW, tps, 5, C_HI);
    draw_str(18, STATUS_ROW, "TOKENS/S", C_DIM);
    draw_num(28, STATUS_ROW, ntok, 6, C_HI);
    draw_str(35, STATUS_ROW, "CHARS", C_DIM);
}

/* RAM log for debuggers / tests: magic, seed, temp, count, tokens */
volatile uint32_t llm_log_hdr[4] __attribute__((aligned(16)));
volatile uint8_t llm_log_tok[4096];

int main(void)
{
    vi_init();
    isv_detect();
    fill(0, H, C_BG);
    draw_str(9, 1, "RETRO-LLM  FOR THE N64", C_HI);
    draw_str(11, 3, "A TINY TRANSFORMER", C_TEXT);
    draw_str(8, 7, "2 LAYERS, 4 HEADS, D=32", C_TEXT);
    draw_str(8, 8, "18688 INT8 PARAMETERS", C_TEXT);
    draw_str(8, 9, "32-CHAR CONTEXT (KV CACHE)", C_TEXT);
    draw_str(8, 10, "TRAINED ON SHAKESPEARE", C_TEXT);
    draw_str(8, 11, "SAME BITS AS NES AND SNES", C_TEXT);
    draw_str(8, 15, "UP/DOWN: TEMPERATURE", C_DIM);
    draw_str(8, 16, "A: NEW STORY", C_DIM);
    draw_str(13, 21, "PRESS START", C_HI);

    /* wait for START (or start by itself after 3 s) */
    uint32_t t0 = count();
    uint16_t prev = 0, pad;
    for (;;) {
        pad = read_pad();
        if ((pad & ~prev) & BTN_START)
            break;
        prev = pad;
        if (count() - t0 > 3 * COUNT_HZ)
            break;
        wait_ticks(COUNT_HZ / 60);
    }
    uint16_t rng = (uint16_t)(count() - t0);
    if (!rng)
        rng = 1;
    int temp = 2;

restart:
    fill(16, H, C_BG);
    isv_str("\nRETRO-LLM SEED=0x");
    isv_hex(rng, 4);
    isv_str(" TEMP=");
    isv_hex((uint32_t)temp, 1);
    isv_str("\n");
    cur_row = 0;
    cur_col = 0;
    llm_reset();
    llm_log_hdr[0] = 0x4C4C4D21;        /* "LLM!" */
    llm_log_hdr[1] = rng;
    llm_log_hdr[2] = (uint32_t)temp;
    llm_log_hdr[3] = 0;
    int tok = 0;
    uint32_t ntok = 0, busy = 0, tps = 0;
    for (;;) {
        int8_t logits[LLM_V];
        uint32_t c0 = count();
        llm_step(tok, logits);
        tok = llm_sample(logits, &rng, temp);
        busy += count() - c0;
        if (ntok < sizeof llm_log_tok)
            llm_log_tok[ntok] = (uint8_t)tok;
        ntok++;
        llm_log_hdr[3] = ntok;
        putc_tok(tok);
        isv_write(tok ? &LLM_VOCAB[tok] : "\n", 1);
        if ((ntok & 15) == 0) {
            tps = (uint32_t)((uint64_t)16 * COUNT_HZ / busy);
            busy = 0;
        }
        draw_status(temp, tps, ntok);

        /* the model runs far faster than anyone reads: type at ~40 chars/s */
        wait_ticks(COUNT_HZ / 40);
        pad = read_pad();
        uint16_t pressed = pad & ~prev;
        prev = pad;
        if ((pressed & BTN_UP) && temp < 3)
            temp++;
        if ((pressed & BTN_DOWN) && temp > 0)
            temp--;
        if (pressed & BTN_A)
            goto restart;
    }
}
