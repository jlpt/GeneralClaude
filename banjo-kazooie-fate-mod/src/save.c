#include "fate.h"
#include "recompconfig.h"

/*
 * Persistence.
 *
 * Banjo-Kazooie: Recompiled fills its whole 2 KB EEPROM image with vanilla
 * and extended save data, so there's no room for an RPG character in it.
 * Instead, at startup the mod moves the game's saves into its own save file
 * (saves/<mod id>/save.bin), and keeps each game file's character in a
 * sibling file (rpg_slot1..3.bin) plus one for heirlooms (rpg_global.bin).
 *
 * recomp_change_save_file flushes pending writes and loads the requested
 * file into the EEPROM buffer, so writing a character is: swap to its file,
 * write the blocks through the game's own EEPROM routine, swap back.
 */

#define MAIN_SAVE_NAME "save"
#define SAVE_BLOCKS(size) (((size) + 7) / 8)

_Static_assert(sizeof(FateSave) <= 2048, "FateSave must fit in one 16K EEPROM image");
_Static_assert(sizeof(FateGlobalSave) <= 2048, "FateGlobalSave must fit in one 16K EEPROM image");

static u8 sIoBuf[2048] __attribute__((aligned(16)));
static bool sGlobalLoaded;
s32 gFateLoadedSlot = -1;

static const char *slot_file(s32 slot) {
    switch (slot) {
        case 0:  return "rpg_slot1";
        case 1:  return "rpg_slot2";
        default: return "rpg_slot3";
    }
}

static void write_file(const char *name, const void *data, u32 size) {
    bzero(sIoBuf, sizeof(sIoBuf));
    bcopy(data, sIoBuf, size);
    recomp_change_save_file(name);
    eeprom_writeBlocks(0, 0, sIoBuf, SAVE_BLOCKS(size));
    recomp_change_save_file(MAIN_SAVE_NAME);
}

static void read_file(const char *name, void *data, u32 size) {
    recomp_change_save_file(name);
    eeprom_readBlocks(0, 0, sIoBuf, SAVE_BLOCKS(size));
    recomp_change_save_file(MAIN_SAVE_NAME);
    bcopy(sIoBuf, data, size);
}

void fate_save_init(void) {
    /* Runs from recomp_on_init, before the game reads any saves. */
    recomp_change_save_file(MAIN_SAVE_NAME);
    gFateLoadedSlot = -1;
}

/* ------------------------------------------------------------------ */
/* Global (heirloom) data                                              */
/* ------------------------------------------------------------------ */

void fate_load_global(void) {
    u32 sum;
    read_file("rpg_global", &gFateGlobal, sizeof(gFateGlobal));
    sum = gFateGlobal.checksum;
    gFateGlobal.checksum = 0;
    if (gFateGlobal.magic != FATE_SAVE_MAGIC || gFateGlobal.version != FATE_SAVE_VERSION
        || gFateGlobal.size != sizeof(gFateGlobal) || fate_checksum(&gFateGlobal, sizeof(gFateGlobal)) != sum) {
        bzero(&gFateGlobal, sizeof(gFateGlobal));
    }
    gFateGlobal.checksum = sum;
    sGlobalLoaded = TRUE;
}

void fate_save_global(void) {
    gFateGlobal.magic = FATE_SAVE_MAGIC;
    gFateGlobal.version = FATE_SAVE_VERSION;
    gFateGlobal.size = sizeof(gFateGlobal);
    gFateGlobal.checksum = 0;
    gFateGlobal.checksum = fate_checksum(&gFateGlobal, sizeof(gFateGlobal));
    write_file("rpg_global", &gFateGlobal, sizeof(gFateGlobal));
}

/* ------------------------------------------------------------------ */
/* Per game file                                                       */
/* ------------------------------------------------------------------ */

void fate_load_slot(s32 slot) {
    static FateSave tmp;
    u32 sum;
    bool valid;

    if (slot < 0 || slot > 2) {
        return;
    }
    if (!sGlobalLoaded) {
        fate_load_global();
    }
    read_file(slot_file(slot), &tmp, sizeof(tmp));
    sum = tmp.checksum;
    tmp.checksum = 0;
    valid = tmp.magic == FATE_SAVE_MAGIC && tmp.version == FATE_SAVE_VERSION && tmp.size == sizeof(FateSave)
            && fate_checksum(&tmp, sizeof(tmp)) == sum && tmp.created;

    gFateLoadedSlot = slot;
    if (valid && gameFile_isNotEmpty(slot)) {
        gFate = tmp;
        fate_seed(gFate.rngState ^ 0xA5A5A5A5);
        fate_recalc();
        gFateHp = gFate.hp > 0 ? (f32)gFate.hp : (f32)gDerived.maxHp;
        gFateMana = (f32)gFate.mana;
        fate_shop_restock();
    } else {
        fate_seed(fate_checksum(&tmp, sizeof(tmp)) ^ (u32)(slot * 7919) ^ (u32)osGetCount());
        fate_new_character();
    }
    fate_hud_mark_dirty();
}

void fate_save_slot(s32 slot) {
    if (slot != gFateLoadedSlot || !gFate.created) {
        return;
    }
    gFate.hp = (u16)fate_clampf(gFateHp, 0.0f, 65535.0f);
    gFate.mana = (u16)fate_clampf(gFateMana, 0.0f, 65535.0f);
    gFate.magic = FATE_SAVE_MAGIC;
    gFate.version = FATE_SAVE_VERSION;
    gFate.size = sizeof(FateSave);
    gFate.checksum = 0;
    gFate.checksum = fate_checksum(&gFate, sizeof(gFate));
    write_file(slot_file(slot), &gFate, sizeof(gFate));
}

void fate_clear_slot(s32 slot) {
    static FateSave empty;
    if (slot < 0 || slot > 2) {
        return;
    }
    bzero(&empty, sizeof(empty));
    write_file(slot_file(slot), &empty, sizeof(empty));
    if (slot == gFateLoadedSlot) {
        gFate.created = FALSE;
        gFateLoadedSlot = -1;
    }
}

/* ------------------------------------------------------------------ */
/* Retirement                                                          */
/* ------------------------------------------------------------------ */

bool fate_can_retire(void) {
    return gFate.created && !gFate.retired && fileProgressFlag_get(FILEPROG_FC_DEFEAT_GRUNTY);
}

void fate_retire(s32 equipSlot) {
    FateItem heir;
    char name[64];
    char msg[128];

    if (!fate_can_retire() || equipSlot < 0 || equipSlot >= SLOT_COUNT || fate_item_empty(&gFate.equip[equipSlot])) {
        return;
    }
    heir = gFate.equip[equipSlot];
    heir.flags |= ITEMF_HEIRLOOM | ITEMF_IDENTIFIED;
    heir.flags &= ~ITEMF_CURSED;

    gFateGlobal.generation = gFate.generation + 1;
    gFateGlobal.hasHeirloom = TRUE;
    gFateGlobal.heirloom = heir;
    bzero(gFateGlobal.ancestorTitle, sizeof(gFateGlobal.ancestorTitle));
    {
        const char *t = fate_fame_title();
        s32 i;
        for (i = 0; i < (s32)sizeof(gFateGlobal.ancestorTitle) - 1 && t[i]; i++) {
            gFateGlobal.ancestorTitle[i] = t[i];
        }
    }
    fate_save_global();
    gFate.retired = TRUE;

    fate_item_name(&heir, name, sizeof(name));
    msg[0] = '\0';
    fate_strcat(msg, "You retire in glory. Your heir will inherit ");
    fate_strcat(msg, name);
    fate_strcat(msg, ". Start a new game file to play as them!");
    fate_banner("RETIRED!", MSG_COLOR_GOLD);
    fate_message(msg, MSG_COLOR_GOLD);
    fate_play_jingle(COMUSIC_2D_PUZZLE_SOLVED_FANFARE);
}
