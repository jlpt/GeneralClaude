#include "fate.h"

/*
 * Developer shortcuts, compiled only with `make DEV=1` for testing and
 * screenshots. Hold L and press:
 *   D-Pad Down   warp to Mumbo's Mountain
 *   D-Pad Left   grant a test kit (all moves, gold, levels, spells, gear, fish)
 */

#ifdef FATE_DEV

void ability_setAllLearned(s32 val);
s32 func_8024E698(s32 controller_index);

static void dev_kit(void) {
    FateItem item;
    s32 r, i;

    ability_setAllLearned(-1); /* a fresh file that skipped Bottles' tutorial knows no moves */
    fate_gain_xp((s32)(fate_xp_for_level(8) - gFate.xp), FALSE);
    gFate.gold += 5000;
    gFate.fame += 600;
    gFate.spellsKnown = (1u << SPELL_COUNT) - 1;
    gFate.poleTier = 2;
    gFate.consumables[CONS_HP_POTION] += 5;
    gFate.consumables[CONS_MANA_POTION] += 5;
    gFate.consumables[CONS_IDENTIFY] += 3;
    gFate.fish[FISH_GLOOP_GUPPY * FISH_SIZE_COUNT + FISH_SIZE_MEDIUM] += 2;
    gFate.fish[FISH_BIGBUTT_BASS * FISH_SIZE_COUNT + FISH_SIZE_SMALL] += 1;
    gFate.gems[2] += 1;
    for (r = RARITY_NORMAL; r <= RARITY_LEGENDARY; r++) {
        for (i = 0; i < 2; i++) {
            fate_item_generate(&item, 4, r, 0);
            if (i == 0) {
                fate_identify(&item);
            }
            fate_inventory_add(&item);
        }
    }
    fate_recalc();
    fate_heal_full();
    fate_message("[dev] Test kit granted.", MSG_COLOR_FAME);
}

/* Let Start skip the new-game intro even when no other save exists. */
RECOMP_PATCH bool cutscene_skipIntroCutsceneCheck(void) {
    return func_8024E698(0) == 1;
}

bool fate_dev_input(s32 dpad[4]) {
    if (dpad[1] == 1) {
        fate_message("[dev] Warping to Mumbo's Mountain.", MSG_COLOR_FAME);
        transitionToMap(MAP_2_MM_MUMBOS_MOUNTAIN, 5, 1);
        return TRUE;
    }
    if (dpad[2] == 1) {
        dev_kit();
        return TRUE;
    }
    return FALSE;
}

#endif
