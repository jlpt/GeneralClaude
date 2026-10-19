#include "fate.h"

/*
 * Fate's fishing. With a pole from Captain Blubber, wade into shallow water
 * (or tread water at the surface), stand still and press L to cast. When the
 * HUD shouts "BITE!", press L again quickly to reel in a fish, which can be
 * sold or fed to your pet to transform it. Deeper worlds hold rarer fish.
 */

enum fishing_state {
    FISH_IDLE,
    FISH_CASTING,
    FISH_WAITING,
    FISH_BITE,
};

static struct {
    u8 state;
    f32 timer;
    char prompt[48];
} sFish;

static bool player_still(void) {
    s32 st = bs_getState();
    return st == BS_1_IDLE || st == BS_2D_SWIM_IDLE;
}

bool fate_can_fish(void) {
    if (!fate_gameplay_active() || gFate.poleTier == 0) {
        return FALSE;
    }
    if (player_getTransformation() != TRANSFORM_1_BANJO || !player_still()) {
        return FALSE;
    }
    if (player_getWaterState() == BSWATERGROUP_1_SURFACE) {
        return TRUE;
    }
    /* wading: feet below a water surface without being deep enough to swim */
    return floor_isCurrentFloorunk59() && func_80294554() && !player_inWater();
}

bool fate_fishing_active(void) {
    return sFish.state != FISH_IDLE;
}

const char *fate_fishing_prompt(void) {
    switch (sFish.state) {
        case FISH_CASTING:
        case FISH_WAITING:
            return "Fishing...  (move to reel in)";
        case FISH_BITE:
            return "BITE!  Press L!";
        default:
            return fate_can_fish() ? "Press L to fish" : "";
    }
}

s32 fate_fish_price(s32 type, s32 size) {
    static const u8 sizeMult[FISH_SIZE_COUNT] = { 1, 2, 4, 12 };
    return gFishDefs[type].basePrice * sizeMult[size];
}

void fate_fish_name(s32 type, s32 size, char *out, s32 cap) {
    out[0] = '\0';
    if (type != FISH_OLD_BOOT) {
        fate_strcat(out, gFishSizeNames[size]);
        fate_strcat(out, " ");
    }
    fate_strcat(out, gFishDefs[type].name);
}

static s32 roll_fish_type(s32 depth) {
    s32 total = 0;
    s32 i;
    s32 r;
    for (i = 0; i < FISH_COUNT; i++) {
        if (gFishDefs[i].minDepth <= depth + 1) {
            total += gFishDefs[i].weight;
        }
    }
    r = fate_rand_range(0, total - 1);
    for (i = 0; i < FISH_COUNT; i++) {
        if (gFishDefs[i].minDepth <= depth + 1) {
            r -= gFishDefs[i].weight;
            if (r < 0) {
                return i;
            }
        }
    }
    return FISH_GLOOP_GUPPY;
}

static s32 roll_fish_size(void) {
    s32 bonus = (gFate.poleTier - 1) * 300;
    s32 r = fate_rand_range(0, 9999) + bonus;
    if (r >= 9500) return FISH_SIZE_FLAWLESS;
    if (r >= 8000) return FISH_SIZE_LARGE;
    if (r >= 5000) return FISH_SIZE_MEDIUM;
    return FISH_SIZE_SMALL;
}

static void catch_something(void) {
    s32 depth = fate_maxi(1, fate_depth());
    s32 r = fate_rand_range(0, 99);
    char msg[96];
    char name[48];

    msg[0] = '\0';
    if (r < 65) {
        s32 type = roll_fish_type(depth);
        s32 size = type == FISH_OLD_BOOT ? FISH_SIZE_SMALL : roll_fish_size();
        s32 idx = type * FISH_SIZE_COUNT + size;
        if (gFate.fish[idx] < 255) {
            gFate.fish[idx]++;
        }
        fate_fish_name(type, size, name, sizeof(name));
        fate_strcat(msg, type == FISH_OLD_BOOT ? "You fished up an " : "You caught a ");
        fate_strcat(msg, name);
        fate_strcat(msg, "!");
        fate_message(msg, size == FISH_SIZE_FLAWLESS ? MSG_COLOR_GOLD : MSG_COLOR_PET);
        fate_play_jingle(size >= FISH_SIZE_LARGE ? COMUSIC_3B_MINIGAME_VICTORY : COMUSIC_2B_DING_B);
    } else if (r < 75) {
        s32 gold = fate_rand_range(5, 20) * depth;
        fate_add_gold(gold);
        fate_strcat(msg, "You hooked a sunken purse: ");
        fate_strcat_int(msg, gold);
        fate_strcat(msg, " gold!");
        fate_message(msg, MSG_COLOR_GOLD);
        fate_play_jingle(COMUSIC_2B_DING_B);
    } else if (r < 85) {
        FateItem item;
        fate_item_generate(&item, depth + 1, -1, gDerived.magicFind + 50);
        fate_item_name(&item, name, sizeof(name));
        if (fate_inventory_add(&item)) {
            fate_strcat(msg, "Something heavy on the line: ");
        } else {
            fate_strcat(msg, "Pack full! It slips back into the water: ");
        }
        fate_strcat(msg, name);
        fate_message(msg, fate_rarity_color(item.rarity));
        fate_play_jingle(COMUSIC_3D_JIGGY_SPAWN);
    } else if (r < 92) {
        s32 gem = fate_rand_range(1, gGemCount - 1);
        gFate.gems[gem]++;
        fate_strcat(msg, "A glint in the water... a ");
        fate_strcat(msg, gGems[gem].name);
        fate_strcat(msg, "!");
        fate_message(msg, 0x7FFFD4FF);
        fate_play_jingle(COMUSIC_2B_DING_B);
    } else {
        gFate.consumables[CONS_HP_POTION]++;
        fate_message("You fished up a corked Health Potion.", MSG_COLOR_DEFAULT);
        fate_play_jingle(COMUSIC_2B_DING_B);
    }
    fate_hud_mark_dirty();
}

void fate_fishing_start(void) {
    if (sFish.state == FISH_BITE) {
        catch_something();
        sFish.state = FISH_IDLE;
        return;
    }
    if (sFish.state != FISH_IDLE || !fate_can_fish()) {
        return;
    }
    sFish.state = FISH_CASTING;
    sFish.timer = 0.6f;
    fate_play_sfx(SFX_F_SMALL_WATER_SPLASH);
    fate_hud_mark_dirty();
}

void fate_fishing_frame(f32 dt) {
    if (sFish.state == FISH_IDLE) {
        return;
    }
    if (!fate_gameplay_active() || !player_still()) {
        if (sFish.state == FISH_BITE) {
            fate_message("It got away...", MSG_COLOR_DEFAULT);
        }
        sFish.state = FISH_IDLE;
        fate_hud_mark_dirty();
        return;
    }
    sFish.timer -= dt;
    if (sFish.timer > 0.0f) {
        return;
    }
    switch (sFish.state) {
        case FISH_CASTING:
            sFish.state = FISH_WAITING;
            sFish.timer = 2.5f + fate_randf() * 4.5f;
            fate_play_sfx(SFX_413_WATER_RIPPLE);
            break;
        case FISH_WAITING:
            sFish.state = FISH_BITE;
            sFish.timer = 0.9f + (f32)gFate.poleTier * 0.15f;
            fate_play_jingle(COMUSIC_2B_DING_B);
            fate_play_sfx(SFX_12_WATER_PADDLING_1);
            break;
        case FISH_BITE:
            fate_message("It got away...", MSG_COLOR_DEFAULT);
            sFish.state = FISH_IDLE;
            break;
    }
    fate_hud_mark_dirty();
}
