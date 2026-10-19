#include "fate.h"
#include "recompconfig.h"
#include "recomputils.h"

/*
 * Entry points: mod init, the per-frame driver, input, and the hooks that
 * tell the RPG layer about map loads, saves, and collectibles.
 *
 * Controls (the D-Pad and L are unused by Banjo-Kazooie: Recompiled):
 *   D-Pad Up     open the Fate menu (character, inventory, pet, spells, quests, Grove shops)
 *   D-Pad Down   quaff a health potion
 *   D-Pad Left   quaff a mana potion
 *   D-Pad Right  cast the selected spell
 *   L            fish when wading in water, otherwise cycle spells
 */

bool gFateMenuOpen;
f32 gFateGameTime;

static bool sUiReady;
static s32 sLastLevel = -1;
static s32 sSavingSlot = -1;
static s32 sZoomboxSlot = -1;
static f32 sDroppedGoldTick;

bool honeycombscore_get(enum honeycomb_e indx);
bool mumboscore_get(enum mumbotoken_e indx);
u32 jiggyscore_isCollected(enum jiggy_e jiggy_id);

/* ------------------------------------------------------------------ */
/* Init                                                                */
/* ------------------------------------------------------------------ */

RECOMP_CALLBACK("*", recomp_on_init) void fate_on_init(void) {
    fate_combat_init();
    fate_save_init();
    recomp_printf("[Fate] Banjo-Kazooie: Fate Edition loaded.\n");
}

/* ------------------------------------------------------------------ */
/* Input                                                               */
/* ------------------------------------------------------------------ */

typedef struct {
    s32 dpad[4];  /* up, down, left, right: frames held */
    s32 side[3];  /* Z, L, R */
} FateButtons;

static void read_buttons(FateButtons *b) {
    func_8024E6E0(0, b->dpad);
    controller_copySideButtons(0, b->side);
}

#ifdef FATE_DEV
bool fate_dev_input(s32 dpad[4]);
#endif

static void handle_input(void) {
    FateButtons b;
    read_buttons(&b);

    /* L+R together is a debug chord that zeroes the pad, so ignore it. */
    if (b.side[1] && b.side[2]) {
        return;
    }
#ifdef FATE_DEV
    if (b.side[1] && fate_dev_input(b.dpad)) {
        return;
    }
#endif
    if (b.dpad[0] == 1) {
        if (!gcdialog_hasCurrentTextId()) {
            fate_menu_open(fate_in_town() ? TAB_GROVE : TAB_CHARACTER);
        }
        return;
    }
    if (b.dpad[1] == 1) {
        fate_drink_health_potion();
    }
    if (b.dpad[2] == 1) {
        fate_drink_mana_potion();
    }
    if (b.dpad[3] == 1) {
        fate_cast_spell(gFate.selectedSpell);
    }
    if (b.side[1] == 1) {
        if (fate_fishing_active() || fate_can_fish()) {
            fate_fishing_start();
        } else {
            fate_cycle_spell();
        }
    }
}

/* ------------------------------------------------------------------ */
/* Per frame                                                           */
/* ------------------------------------------------------------------ */

static void on_level_change(s32 level) {
    FATE_TRACE("level change -> %d (map %d)\n", level, map_get());
    fate_quest_on_level_enter(level);
    if (level == LEVEL_B_SPIRAL_MOUNTAIN) {
        fate_shop_restock();
        fate_pet_heal_full();
        fate_message("Welcome to Grove. Press D-Pad Up to visit the shops and the quest board.", MSG_COLOR_QUEST);
    } else if (fate_depth_for_level(level) > 0) {
        char msg[64];
        msg[0] = '\0';
        fate_strcat(msg, fate_level_name(level));
        fate_strcat(msg, " - Depth ");
        fate_strcat_int(msg, fate_depth_for_level(level));
        fate_banner(msg, MSG_COLOR_DEFAULT);
    }
}

/* Runs at the start of the game's main update, every frame, even while the world is frozen. */
RECOMP_HOOK("func_802E4424") void fate_frame(void) {
    f32 dt = time_getDelta();

    if (dt < 0.0f || dt > 0.25f) {
        dt = 0.0f;
    }
    if (!sUiReady) {
        sUiReady = TRUE;
        fate_hud_init();
        fate_menu_init();
    }

    if (!gFateMenuOpen && fate_gameplay_active()) {
        s32 level = level_get();
        if (level != sLastLevel) {
            sLastLevel = level;
            on_level_change(level);
        }
        gFateGameTime += dt;
        fate_update_regen(dt);
        fate_health_frame(dt);
        fate_combat_frame(dt);
        fate_pet_frame(dt);
        fate_spells_frame(dt);
        fate_fishing_frame(dt);

        if (gFate.droppedGoldSeconds > 0 && map_get() != gFate.droppedGoldMap) {
            sDroppedGoldTick += dt;
            while (sDroppedGoldTick >= 1.0f && gFate.droppedGoldSeconds > 0) {
                sDroppedGoldTick -= 1.0f;
                gFate.droppedGoldSeconds--;
                if (gFate.droppedGoldSeconds == 0 && gFate.droppedGold > 0) {
                    gFate.droppedGold = 0;
                    fate_message("Your dropped gold has been carried off by Gruntlings.", MSG_COLOR_BAD);
                }
            }
        }
        if (!gFateBargainOpen) {
            handle_input();
        }
    }

    fate_menu_frame();
    fate_hud_frame();
}

/* ------------------------------------------------------------------ */
/* Map loading                                                         */
/* ------------------------------------------------------------------ */

/* Registers the pet's actor types every time a map rebuilds its spawn list. */
RECOMP_HOOK_RETURN("spawnQueue_reset") void fate_on_spawn_list_reset(void) {
    FATE_TRACE("spawn list reset, registering pet actors\n");
    fate_pet_register_actors();
}

static s32 sLoadingMap;
static s32 sLoadingExit;

RECOMP_HOOK("func_80334B20") void fate_on_map_load_begin(enum map_e map, s32 exit, s32 arg2) {
    sLoadingMap = map;
    sLoadingExit = exit;
}

RECOMP_HOOK_RETURN("func_80334B20") void fate_on_map_load_end(void) {
    FATE_TRACE("map load %d exit %d\n", sLoadingMap, sLoadingExit);
    if (!gFate.created || gFateLoadedSlot < 0) {
        return;
    }
    fate_combat_on_map_load();
    fate_pet_on_map_load();
    fate_travel_on_map_load(sLoadingMap, sLoadingExit);
}

/* ------------------------------------------------------------------ */
/* Saving and loading                                                  */
/* ------------------------------------------------------------------ */

/* The file select screen loads a game file whenever one is highlighted. */
RECOMP_HOOK("setGameInformationZoombox") void fate_on_zoombox(s32 gamenum) {
    sZoomboxSlot = gamenum;
}

RECOMP_HOOK_RETURN("setGameInformationZoombox") void fate_on_zoombox_return(void) {
    /* Always reload: a reset or game over without saving must not leave stale RPG data around. */
    if (sZoomboxSlot >= 0) {
        fate_load_slot(sZoomboxSlot);
        sLastLevel = -1;
    }
}

/* Every vanilla save of a game file writes its character right after. */
RECOMP_HOOK("gameFile_8033CFD4") void fate_on_game_save(s32 gamenum) {
    sSavingSlot = gamenum;
}

RECOMP_HOOK_RETURN("gameFile_8033CFD4") void fate_on_game_save_return(void) {
    if (sSavingSlot >= 0) {
        fate_save_slot(sSavingSlot);
        sSavingSlot = -1;
    }
}

RECOMP_HOOK("gameFile_clear") void fate_on_game_clear(s32 gamenum) {
    fate_clear_slot(gamenum);
}

/* ------------------------------------------------------------------ */
/* Collectibles                                                        */
/* ------------------------------------------------------------------ */

/* Jiggies are the big prizes of each world: experience, fame and gold. */
RECOMP_HOOK("jiggyscore_setCollected") void fate_on_jiggy(s32 indx, s32 val) {
    s32 depth;
    char msg[64];
    if (!val || !gFate.created || gFateLoadedSlot < 0 || indx <= 0 || indx >= 0x65 || jiggyscore_isCollected(indx)) {
        return;
    }
    depth = fate_maxi(1, fate_depth());
    fate_add_gold(25 * depth);
    msg[0] = '\0';
    fate_strcat(msg, "Jiggy! +");
    fate_strcat_int(msg, 25 * depth);
    fate_strcat(msg, " gold");
    fate_message(msg, MSG_COLOR_GOLD);
    fate_gain_xp(40 + 30 * depth, TRUE);
    fate_gain_fame(12);
    fate_quest_on_jiggy(level_get());
}

RECOMP_HOOK("honeycombscore_set") void fate_on_honeycomb(enum honeycomb_e indx, bool val) {
    if (!val || !gFate.created || gFateLoadedSlot < 0 || honeycombscore_get(indx)) {
        return;
    }
    fate_message("An empty honeycomb piece. Your legend grows.", MSG_COLOR_FAME);
    fate_gain_fame(5);
    fate_recalc();
}

RECOMP_HOOK("mumboscore_set") void fate_on_mumbo_token(enum mumbotoken_e indx, bool val) {
    if (!val || !gFate.created || gFateLoadedSlot < 0 || mumboscore_get(indx)) {
        return;
    }
    fate_add_gold(15);
}

/* Notes are Fate's gold coins: every note found adds to your purse. */
RECOMP_CALLBACK("*", bkrecomp_note_collected_event) void fate_on_note(enum map_e map_id, enum level_e level_id, u8 note_index) {
    if (!gFate.created || gFateLoadedSlot < 0) {
        return;
    }
    fate_add_gold(2);
    fate_quest_on_note(level_id);
}

RECOMP_CALLBACK("*", bkrecomp_dynamic_note_collected_event) void fate_on_dynamic_note(enum map_e map_id, enum level_e level_id) {
    if (!gFate.created || gFateLoadedSlot < 0) {
        return;
    }
    fate_add_gold(2);
    fate_quest_on_note(level_id);
}
