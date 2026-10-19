#include "fate.h"

/* Clang may lower struct/array initialization to memset, which the game doesn't export. */
void *memset(void *dst, int value, unsigned int size) {
    u8 *p = (u8 *)dst;
    while (size--) {
        *p++ = (u8)value;
    }
    return dst;
}

/* xorshift32, seeded per save so loot rolls feel random but saves resume cleanly */
static u32 sRngState = 0x1234567;

void fate_seed(u32 seed) {
    sRngState = seed ? seed : 0x9E3779B9;
}

u32 fate_rand(void) {
    u32 x = sRngState;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    sRngState = x;
    gFate.rngState = x;
    return x;
}

s32 fate_rand_range(s32 lo, s32 hi) {
    if (hi <= lo) {
        return lo;
    }
    return lo + (s32)(fate_rand() % (u32)(hi - lo + 1));
}

f32 fate_randf(void) {
    return (f32)(fate_rand() & 0xFFFFFF) / (f32)0x1000000;
}

bool fate_chance(s32 percentTimes100) {
    return (s32)(fate_rand() % 10000) < percentTimes100;
}

s32 fate_strlen(const char *s) {
    s32 n = 0;
    while (s[n] != '\0') {
        n++;
    }
    return n;
}

char *fate_strcpy(char *dst, const char *src) {
    char *d = dst;
    while ((*d++ = *src++) != '\0') {
    }
    return dst;
}

char *fate_strcat(char *dst, const char *src) {
    fate_strcpy(dst + fate_strlen(dst), src);
    return dst;
}

char *fate_strcat_int(char *dst, s32 value) {
    char tmp[12];
    s32 i = 0;
    u32 v;
    char *d = dst + fate_strlen(dst);

    if (value < 0) {
        *d++ = '-';
        v = (u32)(-value);
    } else {
        v = (u32)value;
    }
    do {
        tmp[i++] = '0' + (v % 10);
        v /= 10;
    } while (v != 0);
    while (i > 0) {
        *d++ = tmp[--i];
    }
    *d = '\0';
    return dst;
}

char *fate_strcat_signed(char *dst, s32 value) {
    if (value >= 0) {
        fate_strcat(dst, "+");
    }
    return fate_strcat_int(dst, value);
}

s32 fate_clampi(s32 v, s32 lo, s32 hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

f32 fate_clampf(f32 v, f32 lo, f32 hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

s32 fate_mini(s32 a, s32 b) {
    return a < b ? a : b;
}

s32 fate_maxi(s32 a, s32 b) {
    return a > b ? a : b;
}

/* FNV-1a */
u32 fate_checksum(const void *data, u32 size) {
    const u8 *p = (const u8 *)data;
    u32 h = 2166136261u;
    while (size--) {
        h ^= *p++;
        h *= 16777619u;
    }
    return h;
}

f32 fate_dist2d(f32 a[3], f32 b[3]) {
    f32 dx = a[0] - b[0];
    f32 dz = a[2] - b[2];
    return gu_sqrtf(dx * dx + dz * dz);
}

f32 fate_dist3d(f32 a[3], f32 b[3]) {
    f32 dx = a[0] - b[0];
    f32 dy = a[1] - b[1];
    f32 dz = a[2] - b[2];
    return gu_sqrtf(dx * dx + dy * dy + dz * dz);
}

void fate_play_sfx(enum sfx_e sfx) {
    func_8030E6D4(sfx);
}

void fate_play_sfx_pitch(enum sfx_e sfx, f32 pitch, s32 volume) {
    func_8030E760(sfx, pitch, volume);
}

void fate_play_jingle(enum comusic_e music) {
    coMusicPlayer_playMusic(music, 20000);
}
