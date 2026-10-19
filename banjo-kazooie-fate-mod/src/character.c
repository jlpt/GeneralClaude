#include "fate.h"
#include "recompconfig.h"

FateSave gFate;
FateGlobalSave gFateGlobal;
FateDerived gDerived;
f32 gFateHp;
f32 gFateMana;

static f32 sRegenAccumHp;
static f32 sRegenAccumMana;

/* ------------------------------------------------------------------ */
/* Character creation                                                  */
/* ------------------------------------------------------------------ */

static void give_starting_item(s32 slot, s32 base) {
    FateItem *it = &gFate.equip[slot];
    fate_item_generate_base(it, base, RARITY_NORMAL, 1);
}

void fate_new_character(void) {
    s32 i;
    s32 gen;

    bzero(&gFate, sizeof(gFate));
    gFate.magic = FATE_SAVE_MAGIC;
    gFate.version = FATE_SAVE_VERSION;
    gFate.size = sizeof(FateSave);
    gFate.created = TRUE;
    gFate.level = 1;

    gen = gFateGlobal.hasHeirloom ? gFateGlobal.generation : 0;
    gFate.generation = gen;

    /* A descendant inherits a little of the ancestor's might. */
    gFate.stats[STAT_STR] = 15 + gen * 2;
    gFate.stats[STAT_DEX] = 15 + gen * 2;
    gFate.stats[STAT_VIT] = 15 + gen * 2;
    gFate.stats[STAT_MAG] = 10 + gen * 2;
    gFate.statPoints = 5;
    gFate.skillPoints = 2 + gen;
    gFate.gold = 100 + gen * 250;

    gFate.consumables[CONS_HP_POTION] = 3;
    gFate.consumables[CONS_MANA_POTION] = 1;
    gFate.consumables[CONS_TOWN_PORTAL] = 2;
    gFate.consumables[CONS_IDENTIFY] = 1;

    give_starting_item(SLOT_BANJO_WEAPON, 1);   /* Honey Claws */
    give_starting_item(SLOT_KAZOOIE_WEAPON, 17); /* Beak Cap */
    give_starting_item(SLOT_ARMOR, 38);          /* Patched Shorts */
    give_starting_item(SLOT_SHIELD, 47);         /* Old Backpack */

    gFate.spellsKnown = 1u << SPELL_HONEY_HEAL;
    gFate.selectedSpell = SPELL_HONEY_HEAL;

    gFate.petSpecies = (u8)recomp_get_config_u32("pet_species");
    if (gFate.petSpecies >= PET_SPECIES_COUNT) {
        gFate.petSpecies = PET_SPECIES_JINJO;
    }
    gFate.petLevel = 1;
    gFate.petForm = (gFate.petSpecies == PET_SPECIES_BIGBUTT) ? PET_FORM_BIGBUTT : PET_FORM_YELLOW_JINJO;
    gFate.petFormPermanent = gFate.petForm;

    if (gFateGlobal.hasHeirloom) {
        FateItem heirloom = gFateGlobal.heirloom;
        heirloom.flags |= ITEMF_HEIRLOOM | ITEMF_IDENTIFIED;
        heirloom.enchant = gen;
        fate_inventory_add(&heirloom);
    }

    gFate.portalMap = 0;
    for (i = 0; i < MAX_ACTIVE_QUESTS; i++) {
        gFate.quests[i].type = QUEST_NONE;
    }
    fate_quests_refresh_board();
    fate_shop_restock();

    fate_recalc();
    gFateHp = (f32)gDerived.maxHp;
    gFateMana = (f32)gDerived.maxMana;
    gFate.petHp = (u16)fate_pet_max_hp();
}

/* ------------------------------------------------------------------ */
/* Experience and fame                                                 */
/* ------------------------------------------------------------------ */

static u32 xp_to_next(s32 level) {
    return 60 + 45 * level + 9 * level * level;
}

u32 fate_xp_for_level(s32 level) {
    u32 total = 0;
    s32 l;
    for (l = 1; l < level; l++) {
        total += xp_to_next(l);
    }
    return total;
}

void fate_gain_xp(s32 amount, bool showMessage) {
    char buf[64];
    s32 levelsGained = 0;

    if (amount <= 0 || !gFate.created) {
        return;
    }
    amount = amount + amount * gDerived.xpBonus / 100;
    gFate.xp += amount;
    fate_pet_gain_xp(amount / 2);

    if (showMessage) {
        buf[0] = '\0';
        fate_strcat(buf, "+");
        fate_strcat_int(buf, amount);
        fate_strcat(buf, " experience");
        fate_message(buf, MSG_COLOR_XP);
    }

    while (gFate.level < FATE_MAX_LEVEL && gFate.xp >= fate_xp_for_level(gFate.level + 1)) {
        gFate.level++;
        gFate.statPoints += STAT_POINTS_PER_LEVEL;
        gFate.skillPoints += SKILL_POINTS_PER_LEVEL;
        levelsGained++;
    }

    if (levelsGained > 0) {
        fate_recalc();
        fate_heal_full();
        buf[0] = '\0';
        fate_strcat(buf, "LEVEL ");
        fate_strcat_int(buf, gFate.level);
        fate_strcat(buf, "!");
        fate_banner(buf, MSG_COLOR_XP);
        fate_message("You feel stronger. Spend your points in the menu (D-Pad Up).", MSG_COLOR_XP);
        fate_play_jingle(COMUSIC_18_HEALTH_UPGRADE);
        fate_play_sfx(SFX_4F_BANJO_WAHOO);
    }
    fate_hud_mark_dirty();
}

u32 fate_fame_for_level(s32 fameLevel) {
    u32 total = 0;
    s32 i;
    for (i = 1; i <= fameLevel; i++) {
        total += 80 + 60 * i;
    }
    return total;
}

s32 fate_fame_level(void) {
    s32 lvl = 0;
    while (lvl < 60 && gFate.fame >= fate_fame_for_level(lvl + 1)) {
        lvl++;
    }
    return lvl;
}

const char *fate_fame_title(void) {
    s32 lvl = fate_fame_level();
    if (lvl >= gFameTitleCount) {
        lvl = gFameTitleCount - 1;
    }
    return gFameTitles[lvl];
}

void fate_gain_fame(s32 amount) {
    char buf[96];
    s32 before;
    s32 after;

    if (amount <= 0 || !gFate.created) {
        return;
    }
    before = fate_fame_level();
    gFate.fame += amount;
    after = fate_fame_level();

    buf[0] = '\0';
    fate_strcat(buf, "+");
    fate_strcat_int(buf, amount);
    fate_strcat(buf, " fame");
    fate_message(buf, MSG_COLOR_FAME);

    if (after > before) {
        gFate.skillPoints += (after - before) * SKILL_POINTS_PER_FAME;
        buf[0] = '\0';
        fate_strcat(buf, "Your fame grows! You are now known as ");
        fate_strcat(buf, fate_fame_title());
        fate_banner("FAME RISES!", MSG_COLOR_FAME);
        fate_message(buf, MSG_COLOR_FAME);
        fate_play_jingle(COMUSIC_42_NOTEDOOR_OPENING_FANFARE);
    }
    fate_hud_mark_dirty();
}

void fate_add_gold(s32 amount) {
    if (amount <= 0) {
        return;
    }
    gFate.gold += amount;
    if (gFate.gold > 99999999) {
        gFate.gold = 99999999;
    }
    fate_hud_mark_dirty();
}

bool fate_spend_gold(s32 amount) {
    if (amount < 0 || gFate.gold < (u32)amount) {
        fate_message("You can't afford that.", MSG_COLOR_BAD);
        fate_play_jingle(COMUSIC_2C_BUZZER);
        return FALSE;
    }
    gFate.gold -= amount;
    fate_hud_mark_dirty();
    return TRUE;
}

bool fate_raise_stat(s32 stat) {
    if (gFate.statPoints == 0 || stat < 0 || stat >= STAT_COUNT || gFate.stats[stat] >= 999) {
        return FALSE;
    }
    gFate.statPoints--;
    gFate.stats[stat]++;
    fate_recalc();
    return TRUE;
}

bool fate_raise_skill(s32 skill) {
    if (gFate.skillPoints == 0 || skill < 0 || skill >= SKILL_COUNT || gFate.skills[skill] >= FATE_SKILL_MAX) {
        return FALSE;
    }
    gFate.skillPoints--;
    gFate.skills[skill]++;
    fate_recalc();
    return TRUE;
}

/* ------------------------------------------------------------------ */
/* Derived stats                                                       */
/* ------------------------------------------------------------------ */

static s32 equip_effect(s32 effect) {
    s32 total = 0;
    s32 i;
    for (i = 0; i < SLOT_COUNT; i++) {
        total += fate_item_effect_total(&gFate.equip[i], effect);
    }
    return total;
}

/* Base numbers scale gently with how far an item's level exceeds its tier. */
static s32 scaled_base(const FateItem *it, s32 value) {
    const FateItemBase *b = fate_item_base(it);
    s32 over = (s32)it->ilvl - (s32)b->tier;
    s32 pct = 100;
    if (over > 0) {
        pct += over * 5;
    }
    if (it->rarity == RARITY_ELITE) {
        pct += 50;
    } else if (it->rarity == RARITY_LEGENDARY) {
        pct += 100;
    }
    if (it->flags & ITEMF_HEIRLOOM) {
        pct += 25 * it->enchant;
    }
    return value * pct / 100;
}

void fate_recalc(void) {
    FateDerived *d = &gDerived;
    s32 i;
    s32 allStats = equip_effect(FX_ALL_STATS);
    s32 allSkills = equip_effect(FX_ALL_SKILLS);
    s32 armor = 0;
    s32 honeyBonus;
    const FateItem *w;

    bzero(d, sizeof(*d));

    d->stat[STAT_STR] = gFate.stats[STAT_STR] + equip_effect(FX_STR) + allStats;
    d->stat[STAT_DEX] = gFate.stats[STAT_DEX] + equip_effect(FX_DEX) + allStats;
    d->stat[STAT_VIT] = gFate.stats[STAT_VIT] + equip_effect(FX_VIT) + allStats;
    d->stat[STAT_MAG] = gFate.stats[STAT_MAG] + equip_effect(FX_MAG) + allStats;
    for (i = 0; i < SKILL_COUNT; i++) {
        d->skill[i] = gFate.skills[i] + (gFate.skills[i] > 0 || allSkills > 0 ? allSkills : 0);
    }

    /* Empty honeycomb upgrades from vanilla still matter: each adds 6% health. */
    honeyBonus = (D_80385F30[ITEM_15_HEALTH_TOTAL] - 5) * 6;
    if (honeyBonus < 0) {
        honeyBonus = 0;
    }
    d->maxHp = 40 + 5 * d->stat[STAT_VIT] + 4 * gFate.level + equip_effect(FX_HP);
    d->maxHp = d->maxHp + d->maxHp * honeyBonus / 100;
    d->maxMana = 10 + 3 * d->stat[STAT_MAG] + 2 * gFate.level + equip_effect(FX_MANA);

    for (i = SLOT_HELM; i <= SLOT_BOOTS; i++) {
        const FateItem *it = &gFate.equip[i];
        if (!fate_item_empty(it)) {
            armor += scaled_base(it, fate_item_base(it)->a);
        }
    }
    d->armor = armor + armor * equip_effect(FX_ARMOR_PCT) / 100 + equip_effect(FX_ARMOR_FLAT) + d->stat[STAT_DEX] / 5;

    w = &gFate.equip[SLOT_BANJO_WEAPON];
    if (!fate_item_empty(w)) {
        d->pawMin = scaled_base(w, fate_item_base(w)->a);
        d->pawMax = scaled_base(w, fate_item_base(w)->b);
    } else {
        d->pawMin = 1;
        d->pawMax = 3;
    }
    w = &gFate.equip[SLOT_KAZOOIE_WEAPON];
    if (!fate_item_empty(w)) {
        d->beakMin = scaled_base(w, fate_item_base(w)->a);
        d->beakMax = scaled_base(w, fate_item_base(w)->b);
    } else {
        d->beakMin = 1;
        d->beakMax = 2;
    }

    d->dmgPct = equip_effect(FX_DMG_PCT);
    d->dmgFlat = equip_effect(FX_DMG_FLAT);
    d->eggPct = equip_effect(FX_EGG_DMG);
    d->beakPct = equip_effect(FX_BEAK_DMG);
    d->pawPct = equip_effect(FX_PAW_DMG);
    d->critChance = 500 + d->stat[STAT_DEX] * 15 + d->skill[SKILL_CRIT] * 100 + equip_effect(FX_CRIT) * 100;
    d->critChance = fate_mini(d->critChance, 6000);
    d->critMult = 200 + d->skill[SKILL_CRIT] * 5;
    if (!fate_item_empty(&gFate.equip[SLOT_SHIELD])) {
        d->block = fate_mini(5 + d->skill[SKILL_SHIELD] * 2 + equip_effect(FX_BLOCK), 50);
    }
    d->dodge = fate_mini(d->stat[STAT_DEX] / 4 + equip_effect(FX_DODGE), 40);
    d->lifeSteal = equip_effect(FX_LIFE_STEAL);
    d->goldFind = equip_effect(FX_GOLD_FIND);
    d->magicFind = equip_effect(FX_MAGIC_FIND);
    d->xpBonus = equip_effect(FX_XP);
    d->hpRegen = equip_effect(FX_HP_REGEN);
    d->manaRegen = equip_effect(FX_MANA_REGEN);
    d->petDmgPct = equip_effect(FX_PET_DMG);
    d->petHpPct = equip_effect(FX_PET_HP);
    d->spellPct = equip_effect(FX_SPELL_PWR) + d->stat[STAT_MAG] * 2;
    d->moveSpeed = fate_mini(equip_effect(FX_MOVE_SPEED), 40);
    d->thorns = equip_effect(FX_THORNS);
    d->fireDmg = equip_effect(FX_FIRE_DMG);
    d->dmgReduce = equip_effect(FX_DMG_REDUCE);

    if (gFateHp > (f32)d->maxHp) {
        gFateHp = (f32)d->maxHp;
    }
    if (gFateMana > (f32)d->maxMana) {
        gFateMana = (f32)d->maxMana;
    }
    fate_hud_mark_dirty();
}

void fate_heal_full(void) {
    gFateHp = (f32)gDerived.maxHp;
    gFateMana = (f32)gDerived.maxMana;
    fate_health_sync_to_bk();
    fate_hud_mark_dirty();
}

void fate_update_regen(f32 dt) {
    f32 hpPerSec = 0.15f + gDerived.stat[STAT_VIT] * 0.02f + gDerived.hpRegen * 0.1f;
    f32 manaPerSec = 0.4f + gDerived.stat[STAT_MAG] * 0.05f + gDerived.manaRegen * 0.1f;

    if (fate_in_town()) {
        hpPerSec *= 5.0f;
        manaPerSec *= 3.0f;
    }
    if (gFateHp < (f32)gDerived.maxHp) {
        gFateHp = fate_clampf(gFateHp + hpPerSec * dt, 0.0f, (f32)gDerived.maxHp);
        sRegenAccumHp += dt;
        if (sRegenAccumHp > 0.5f) {
            sRegenAccumHp = 0.0f;
            fate_health_sync_to_bk();
        }
    }
    if (gFateMana < (f32)gDerived.maxMana) {
        gFateMana = fate_clampf(gFateMana + manaPerSec * dt, 0.0f, (f32)gDerived.maxMana);
    }
    sRegenAccumMana += dt;
}

/* ------------------------------------------------------------------ */
/* Depth                                                               */
/* ------------------------------------------------------------------ */

s32 fate_depth_for_level(enum level_e level) {
    s32 base;
    switch (level) {
        case LEVEL_1_MUMBOS_MOUNTAIN:     base = 1; break;
        case LEVEL_2_TREASURE_TROVE_COVE: base = 2; break;
        case LEVEL_3_CLANKERS_CAVERN:     base = 3; break;
        case LEVEL_4_BUBBLEGLOOP_SWAMP:   base = 4; break;
        case LEVEL_5_FREEZEEZY_PEAK:      base = 5; break;
        case LEVEL_7_GOBIS_VALLEY:        base = 6; break;
        case LEVEL_A_MAD_MONSTER_MANSION: base = 7; break;
        case LEVEL_9_RUSTY_BUCKET_BAY:    base = 8; break;
        case LEVEL_8_CLICK_CLOCK_WOOD:    base = 9; break;
        case LEVEL_C_BOSS:                base = 10; break;
        case LEVEL_6_LAIR:
            base = 1 + fate_mini(8, jiggyscore_total() / 10);
            break;
        default:
            return 0; /* Grove and cutscenes are safe */
    }
    return base + gFate.nightmare * 10 + gFate.generation * 2;
}

s32 fate_depth(void) {
    return fate_depth_for_level(level_get());
}

const char *fate_level_name(enum level_e level) {
    switch (level) {
        case LEVEL_1_MUMBOS_MOUNTAIN:     return "Mumbo's Mountain";
        case LEVEL_2_TREASURE_TROVE_COVE: return "Treasure Trove Cove";
        case LEVEL_3_CLANKERS_CAVERN:     return "Clanker's Cavern";
        case LEVEL_4_BUBBLEGLOOP_SWAMP:   return "Bubblegloop Swamp";
        case LEVEL_5_FREEZEEZY_PEAK:      return "Freezeezy Peak";
        case LEVEL_6_LAIR:                return "Gruntilda's Lair";
        case LEVEL_7_GOBIS_VALLEY:        return "Gobi's Valley";
        case LEVEL_8_CLICK_CLOCK_WOOD:    return "Click Clock Wood";
        case LEVEL_9_RUSTY_BUCKET_BAY:    return "Rusty Bucket Bay";
        case LEVEL_A_MAD_MONSTER_MANSION: return "Mad Monster Mansion";
        case LEVEL_B_SPIRAL_MOUNTAIN:     return "Grove (Spiral Mountain)";
        case LEVEL_C_BOSS:                return "Gruntilda's Tower";
        default:                          return "";
    }
}

bool fate_in_town(void) {
    return level_get() == LEVEL_B_SPIRAL_MOUNTAIN;
}

bool fate_gameplay_active(void) {
    s32 lvl;
    if (!gFate.created || gFateLoadedSlot < 0) {
        return FALSE;
    }
    if (getGameMode() != GAME_MODE_3_NORMAL) {
        return FALSE;
    }
    lvl = level_get();
    if (lvl <= 0 || lvl >= LEVEL_D_CUTSCENE) {
        return FALSE;
    }
    /* The file select screen is filed under Spiral Mountain. */
    if (map_get() == MAP_91_FILE_SELECT) {
        return FALSE;
    }
    return player_is_present();
}
