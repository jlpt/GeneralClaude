#include "fate.h"

/*
 * Fate's town portals. A Scroll of Return (or the Warp Home spell) takes
 * Banjo to Grove and remembers where he left; from Grove he can step back
 * through the portal to the entrance of that map.
 */

#define GROVE_MAP  MAP_1_SM_SPIRAL_MOUNTAIN
#define GROVE_EXIT 0x12 /* in front of Banjo's house */

bool fate_can_warp(void) {
    s32 lvl = level_get();
    if (!fate_gameplay_active() || player_isDead() || gcdialog_hasCurrentTextId()) {
        return FALSE;
    }
    /* No escaping the final battle or Furnace Fun. */
    if (lvl == LEVEL_C_BOSS || map_get() == MAP_8E_GL_FURNACE_FUN) {
        return FALSE;
    }
    return TRUE;
}

bool fate_warp_to_town(void) {
    if (fate_in_town()) {
        fate_message("You're already in Grove.", MSG_COLOR_DEFAULT);
        return FALSE;
    }
    if (!fate_can_warp()) {
        fate_message("The magic fizzles. You can't leave from here.", MSG_COLOR_BAD);
        return FALSE;
    }
    gFate.portalMap = (u8)map_get();
    gFate.portalExit = (u8)exit_get();
    fate_play_jingle(COMUSIC_69_FF_WARP);
    transitionToMap(GROVE_MAP, GROVE_EXIT, 1);
    return TRUE;
}

bool fate_warp_to_portal(void) {
    s32 map = gFate.portalMap;
    if (map == 0) {
        fate_message("There's no portal to return through.", MSG_COLOR_DEFAULT);
        return FALSE;
    }
    if (!fate_in_town() || !fate_can_warp()) {
        return FALSE;
    }
    gFate.portalMap = 0;
    fate_play_jingle(COMUSIC_69_FF_WARP);
    transitionToMap(map, gFate.portalExit, 1);
    return TRUE;
}

void fate_travel_on_map_load(enum map_e map, s32 exit) {
    char msg[64];
    if (gFate.droppedGold > 0 && gFate.droppedGoldSeconds > 0 && map == gFate.droppedGoldMap) {
        msg[0] = '\0';
        fate_strcat(msg, "You recover your dropped gold: ");
        fate_strcat_int(msg, (s32)gFate.droppedGold);
        fate_message(msg, MSG_COLOR_GOLD);
        fate_add_gold((s32)gFate.droppedGold);
        gFate.droppedGold = 0;
        gFate.droppedGoldSeconds = 0;
        fate_play_jingle(COMUSIC_3B_MINIGAME_VICTORY);
    }
}
