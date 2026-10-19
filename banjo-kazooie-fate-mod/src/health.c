#include "fate.h"
#include "recompconfig.h"

/*
 * Fate-style hit points layered on top of Banjo's honeycomb health.
 *
 * Banjo keeps a real HP pool (gFateHp). Every change to the vanilla health
 * counter goes through item_adjustByDiff, which this file replaces: a hit
 * that would cost N honeycombs instead deals N x (depth damage) minus armor,
 * dodge and block, and the honeycomb meter is redrawn as a fraction of the
 * HP pool. Vanilla only sees health reach zero when Fate HP does, or when a
 * scripted instant death (pits, drowning, the Furnace Fun death square)
 * calls item_set(HEALTH, 0).
 */

bool gFateBargainPending;
bool gFateBargainOpen;

static s32 sInItemSet;
static s32 sInEnemyContact;
static ActorMarker *sContactMarker;
static s32 sBypassFilter;
static s32 sLastBkHealth = -1;
static ActorMarker *sThornsTarget;
static s32 sThornsDamage;

enum bs_e func_802962BC(u32 arg0);

/* ------------------------------------------------------------------ */
/* Mapping between HP and honeycombs                                   */
/* ------------------------------------------------------------------ */

static s32 bk_total(void) {
    s32 total = D_80385F30[ITEM_15_HEALTH_TOTAL];
    return total > 0 ? total : 5;
}

static s32 mapped_count(void) {
    s32 total = bk_total();
    s32 c;
    if (gFateHp <= 0.0f || gDerived.maxHp <= 0) {
        return 0;
    }
    c = (s32)(gFateHp * (f32)total / (f32)gDerived.maxHp + 0.999f);
    return fate_clampi(c, 1, total);
}

static bool filter_active(void) {
    return gFate.created && gFateLoadedSlot >= 0 && sBypassFilter == 0 && !func_802E4A08();
}

void fate_health_sync_to_bk(void) {
    if (!filter_active() || D_80385F30[ITEM_14_HEALTH] == 0) {
        return; /* never resurrect a player the game already considers dead */
    }
    D_80385F30[ITEM_14_HEALTH] = mapped_count();
    sLastBkHealth = D_80385F30[ITEM_14_HEALTH];
}

/* ------------------------------------------------------------------ */
/* Incoming damage                                                     */
/* ------------------------------------------------------------------ */

static s32 incoming_damage(s32 segments) {
    s32 per = fate_monster_damage_scale();
    s32 depth = fate_depth();
    f32 dmg;
    s32 reduction;
    FateEnemy *e = NULL;

    if (per <= 0) {
        per = 6; /* Grove's training vegetables still sting a little */
    }
    dmg = (f32)(per * segments);

    if (sInEnemyContact && sContactMarker != NULL) {
        e = fate_enemy_get(sContactMarker);
        if (e != NULL && e->champion != CHAMP_NONE) {
            dmg *= (e->champion == CHAMP_BRUTAL) ? 2.0f : 1.4f;
        }
        if (fate_chance(gDerived.dodge * 100)) {
            fate_message("Dodged!", MSG_COLOR_XP);
            return 0;
        }
        if (fate_chance(gDerived.block * 100)) {
            dmg *= 0.25f;
            fate_message("Blocked!", MSG_COLOR_XP);
        }
        if (gDerived.thorns > 0) {
            /* Applied next frame: we're in the middle of collision handling. */
            sThornsTarget = sContactMarker;
            sThornsDamage = gDerived.thorns;
        }
    }

    reduction = gDerived.armor * 100 / (gDerived.armor + 50 + 15 * fate_maxi(depth, 1));
    reduction = fate_mini(reduction, 75);
    dmg = dmg * (f32)(100 - reduction) / 100.0f - (f32)gDerived.dmgReduce;
    if (gAuraTimer > 0.0f) {
        dmg *= 0.5f;
    }
    if (dmg < 1.0f) {
        dmg = 1.0f;
    }
    if (e != NULL && e->champion == CHAMP_VAMPIRIC) {
        e->hp = fate_mini(e->maxHp, e->hp + (s32)dmg);
    }
    return (s32)dmg;
}

/* Returns the vanilla diff to apply so the honeycomb meter mirrors Fate HP. */
static s32 filter_health_diff(s32 oldCount, s32 diff) {
    s32 total = bk_total();
    f32 maxHp = (f32)gDerived.maxHp;

    if (!filter_active() || diff == 0) {
        return diff;
    }

    if (sInItemSet > 0) {
        s32 target = oldCount + diff;
        if (target <= 0) {
            gFateHp = 0.0f; /* scripted instant death */
            return diff;
        }
        if (target >= total) {
            gFateHp = maxHp;
        } else if (gFateHp < maxHp * (f32)target / (f32)total) {
            gFateHp = maxHp * (f32)target / (f32)total;
        }
        return mapped_count() - oldCount;
    }

    if (diff > 0) {
        if (diff >= 9999) {
            gFateHp = maxHp;
        } else {
            /* A honeycomb is a healing draught: 20% of max health per piece. */
            gFateHp = fate_clampf(gFateHp + maxHp * 0.20f * (f32)diff + 5.0f, 0.0f, maxHp);
        }
        fate_hud_mark_dirty();
        return mapped_count() - oldCount;
    }

    gFateHp -= (f32)incoming_damage(-diff);
    fate_hud_mark_dirty();
    if (gFateHp <= 0.0f) {
        if (recomp_get_config_u32("death_bargain") == 0 && !gFateBargainPending && fate_depth() > 0) {
            /* Gruntilda offers a bargain instead of a death. Hold at one honeycomb. */
            gFateHp = 1.0f;
            gFateBargainPending = TRUE;
            return 1 - oldCount;
        }
        gFateHp = 0.0f;
        return -oldCount;
    }
    return mapped_count() - oldCount;
}

/* ------------------------------------------------------------------ */
/* Hooks                                                               */
/* ------------------------------------------------------------------ */

RECOMP_HOOK("item_set") void fate_hook_item_set(s32 item, s32 val) {
    sInItemSet++;
}

RECOMP_HOOK_RETURN("item_set") void fate_hook_item_set_return(void) {
    if (sInItemSet > 0) {
        sInItemSet--;
    }
}

/* Enemy contact: remember who hit Banjo so champions, dodge and block apply. */
RECOMP_HOOK("baMarker_8028D7B8") void fate_hook_enemy_contact(s32 arg0, ActorMarker *arg1, struct5Cs *collision_flags) {
    sInEnemyContact = 1;
    sContactMarker = arg1;
}

RECOMP_HOOK_RETURN("baMarker_8028D7B8") void fate_hook_enemy_contact_return(void) {
    sInEnemyContact = 0;
    sContactMarker = NULL;
}

/*
 * @fate Replaces item_adjustByDiff (code_BEF20.c). Identical to vanilla
 * except for the health filter at the top.
 */
RECOMP_PATCH s32 item_adjustByDiff(enum item_e item, s32 diff, s32 no_hud) {
    s32 oldVal;
    s32 sp3C;
    s32 sp38;
    s32 sp34;
    s32 sp30;
    s32 sp2C;
    s32 sp28;

    oldVal = D_80385F30[item];

    if (ucode_stub3())
        diff = 0;

    /* @fate */
    if (item == ITEM_14_HEALTH) {
        diff = filter_health_diff(oldVal, diff);
    }

    if (diff < 0) {
        if ((item == ITEM_D_EGGS && volatileFlag_get(VOLATILE_FLAG_74_SANDCASTLE_INFINITE_EGGS))
            || (item == ITEM_16_LIFE && volatileFlag_get(VOLATILE_FLAG_73_SANDCASTLE_INFINITE_LIVES))
            || (item == ITEM_F_RED_FEATHER && volatileFlag_get(VOLATILE_FLAG_75_SANDCASTLE_INFINITE_RED_FEATHERS))
            || (item == ITEM_10_GOLD_FEATHER && volatileFlag_get(VOLATILE_FLAG_76_SANDCASTLE_INFINITE_GOLD_FEATHERS))
            || (item == ITEM_17_AIR && volatileFlag_get(VOLATILE_FLAG_96_SANDCASTLE_INFINITE_AIR))
        ) {
            diff = 0;
        }
    }

    sp28 = D_80385F30[item] = MAX(0, D_80385F30[item] + diff);

    sp34 = ((fileProgressFlag_get(FILEPROG_B9_DOUBLE_HEALTH)) ? 2 : 1);
    D_80385F30[ITEM_15_HEALTH_TOTAL] = MIN(sp34 * 8, D_80385F30[ITEM_15_HEALTH_TOTAL]);
    D_80385F30[ITEM_14_HEALTH] = MIN(D_80385F30[ITEM_15_HEALTH_TOTAL], D_80385F30[ITEM_14_HEALTH]);
    D_80385F30[ITEM_17_AIR] = MIN(3600, D_80385F30[ITEM_17_AIR]);
    D_80385F30[ITEM_25_MUMBO_TOKEN_TOTAL] = D_80385F30[ITEM_1C_MUMBO_TOKEN];
    D_80385F30[ITEM_16_LIFE] = MIN(0xFF, D_80385F30[ITEM_16_LIFE]);

    switch (item) {
        case ITEM_D_EGGS:
            sp38 = (fileProgressFlag_get(FILEPROG_BE_CHEATO_BLUEEGGS)) ? 200 : 100;
            break;

        case ITEM_F_RED_FEATHER:
            sp38 = (fileProgressFlag_get(FILEPROG_BF_CHEATO_REDFEATHERS)) ? 100 : 50;
            break;

        case ITEM_10_GOLD_FEATHER:
            sp38 = (fileProgressFlag_get(FILEPROG_C0_CHEATO_GOLDFEATHERS)) ? 20 : 10;
            break;

        default:
            sp38 = 0;
            break;
    }
    if (sp38) {
        D_80385F30[item] = MIN(sp38, D_80385F30[item]);
    }
    if (!no_hud) {
        func_802FACA4(item);
        if (item == ITEM_14_HEALTH || item == ITEM_17_AIR)
            func_802FACA4(ITEM_16_LIFE);
    }

    sp3C = item_empty(item);
    if (item < 6 && sp3C)
        D_80385F30[item + ITEM_6_HOURGLASS] = 0;

    switch (item) {
        case ITEM_14_HEALTH:
            if (sp3C)
                D_80385FE4 = 1;
            if (filter_active()) {
                sLastBkHealth = D_80385F30[ITEM_14_HEALTH];
            }
            break;
        case ITEM_17_AIR:
            sp30 = fxairscore_time_to_count(oldVal);
            sp2C = fxairscore_time_to_count(sp28);
            if (sp3C) {
                bs_checkInterrupt(0x11);
                D_80385FE4 = 1;
            }
            if (sp2C && sp30 != sp2C) {
                if (sp2C < sp30) {
                    coMusicPlayer_playMusic(SFX_AIR_METER_DROPPING, 28000);
                } else {
                    func_8030E760(0x3e9, 1.2f, 28000);
                }
            }
            break;
        case ITEM_C_NOTE:
            sp28 = itemscore_noteScores_get(level_get());
            func_80346DB4(D_80385F30[item]);
            if (D_80385F30[item] == 100 && sp28 != 100) {
                coMusicPlayer_playMusic(COMUSIC_36_100TH_NOTE_COLLECTED, 20000);
                item_inc(ITEM_16_LIFE);
            }
            break;
        case ITEM_26_JIGGY_TOTAL:
            D_80385F30[ITEM_2B_UNKNOWN] += diff;
            break;
        default:
            break;
    }
    return D_80385F30[item];
}

/* ------------------------------------------------------------------ */
/* Per frame                                                           */
/* ------------------------------------------------------------------ */

void fate_health_frame(f32 dt) {
    s32 cur = D_80385F30[ITEM_14_HEALTH];

    /* Vanilla refills health directly after a respawn or at refill pillows. */
    if (sLastBkHealth >= 0 && cur > sLastBkHealth && cur >= bk_total()) {
        gFateHp = (f32)gDerived.maxHp;
    }
    if (cur == 0) {
        gFateHp = 0.0f;
    } else if (gFateHp <= 0.0f) {
        /* Respawned after a vanilla death. */
        gFateHp = (f32)gDerived.maxHp;
        gFateMana = (f32)gDerived.maxMana;
    }
    sLastBkHealth = cur;

    if (sThornsTarget != NULL) {
        ActorMarker *m = sThornsTarget;
        s32 dmg = sThornsDamage;
        sThornsTarget = NULL;
        fate_damage_enemy(marker_getActor(m), dmg, TRUE, FALSE, NULL);
    }

    if (gFateBargainPending && !gFateBargainOpen && fate_gameplay_active() && !player_isDead()) {
        fate_bargain_open();
    }
}

/* ------------------------------------------------------------------ */
/* Potions                                                             */
/* ------------------------------------------------------------------ */

void fate_drink_health_potion(void) {
    f32 maxHp = (f32)gDerived.maxHp;
    f32 missing = maxHp - gFateHp;
    s32 cons;
    f32 heal;
    char msg[64];

    if (gFateHp <= 0.0f) {
        return;
    }
    if (missing < 1.0f) {
        fate_message("You're already at full health.", MSG_COLOR_DEFAULT);
        return;
    }
    if (gFate.consumables[CONS_HP_POTION] > 0 && (missing <= maxHp * 0.45f || gFate.consumables[CONS_HP_POTION_GREATER] == 0)) {
        cons = CONS_HP_POTION;
        heal = maxHp * 0.35f + 20.0f;
    } else if (gFate.consumables[CONS_HP_POTION_GREATER] > 0) {
        cons = CONS_HP_POTION_GREATER;
        heal = maxHp * 0.70f + 50.0f;
    } else {
        fate_message("No health potions! Buy some at Bottles' Burrow.", MSG_COLOR_BAD);
        fate_play_jingle(COMUSIC_2C_BUZZER);
        return;
    }
    gFate.consumables[cons]--;
    gFateHp = fate_clampf(gFateHp + heal, 0.0f, maxHp);
    fate_health_sync_to_bk();
    fate_play_jingle(COMUSIC_16_HONEYCOMB_COLLECTED);
    msg[0] = '\0';
    fate_strcat(msg, "Drank a ");
    fate_strcat(msg, gConsumableNames[cons]);
    fate_message(msg, MSG_COLOR_XP);
    fate_hud_mark_dirty();
}

void fate_drink_mana_potion(void) {
    f32 maxMana = (f32)gDerived.maxMana;
    f32 missing = maxMana - gFateMana;
    s32 cons;
    f32 gain;

    if (missing < 1.0f) {
        fate_message("Your mana is already full.", MSG_COLOR_DEFAULT);
        return;
    }
    if (gFate.consumables[CONS_MANA_POTION] > 0 && (missing <= maxMana * 0.5f || gFate.consumables[CONS_MANA_POTION_GREATER] == 0)) {
        cons = CONS_MANA_POTION;
        gain = maxMana * 0.40f + 10.0f;
    } else if (gFate.consumables[CONS_MANA_POTION_GREATER] > 0) {
        cons = CONS_MANA_POTION_GREATER;
        gain = maxMana * 0.80f + 30.0f;
    } else {
        fate_message("No mana potions!", MSG_COLOR_BAD);
        fate_play_jingle(COMUSIC_2C_BUZZER);
        return;
    }
    gFate.consumables[cons]--;
    gFateMana = fate_clampf(gFateMana + gain, 0.0f, maxMana);
    fate_play_jingle(COMUSIC_14_GOLD_FEATHER_COLLECTED);
    fate_hud_mark_dirty();
}

/* ------------------------------------------------------------------ */
/* Gruntilda's bargain (Fate's death choices)                          */
/* ------------------------------------------------------------------ */

void fate_bargain_resolve(s32 choice) {
    char msg[96];
    gFateBargainPending = FALSE;
    gFate.deaths++;

    switch (choice) {
        case 0: { /* revive on the spot, lose experience and fame */
            u32 floor = fate_xp_for_level(gFate.level);
            u32 loss = (gFate.xp - floor) / 10 + (fate_xp_for_level(gFate.level + 1) - floor) / 20;
            u32 fameLoss = gFate.fame / 20;
            gFate.xp = (gFate.xp - floor > loss) ? gFate.xp - loss : floor;
            gFate.fame -= fameLoss;
            fate_heal_full();
            msg[0] = '\0';
            fate_strcat(msg, "Revived. Lost ");
            fate_strcat_int(msg, (s32)loss);
            fate_strcat(msg, " experience and ");
            fate_strcat_int(msg, (s32)fameLoss);
            fate_strcat(msg, " fame.");
            fate_message(msg, MSG_COLOR_BAD);
            break;
        }
        case 1: { /* revive at the entrance, lose a tenth of your gold */
            u32 loss = gFate.gold / 10;
            gFate.gold -= loss;
            fate_heal_full();
            msg[0] = '\0';
            fate_strcat(msg, "Gruntilda takes ");
            fate_strcat_int(msg, (s32)loss);
            fate_strcat(msg, " gold.");
            fate_message(msg, MSG_COLOR_BAD);
            transitionToMap(map_get(), exit_get(), 1);
            break;
        }
        case 2: { /* flee to Grove, dropping your gold where you fell */
            gFate.droppedGold = gFate.gold;
            gFate.droppedGoldMap = (u8)map_get();
            gFate.droppedGoldSeconds = 300;
            gFate.gold = 0;
            fate_heal_full();
            fate_message("You flee to Grove. Your gold lies where you fell for 5 minutes!", MSG_COLOR_BAD);
            transitionToMap(MAP_1_SM_SPIRAL_MOUNTAIN, 0x12, 1);
            break;
        }
        default: { /* accept your fate: a vanilla death */
            gFateHp = 0.0f;
            sBypassFilter++;
            item_set(ITEM_14_HEALTH, 0);
            sBypassFilter--;
            bs_setState(func_802962BC(0));
            break;
        }
    }
    fate_hud_mark_dirty();
}
