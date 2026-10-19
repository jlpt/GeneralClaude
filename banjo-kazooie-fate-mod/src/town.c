#include "fate.h"

/*
 * Grove's merchants (Spiral Mountain):
 *   Bottles' Burrow       - potions, scrolls, identification, buys anything
 *   Gobi's Arms & Armor   - a rotating stock of weapons and armor
 *   Cheato's Spellbooks   - spells
 *   Mumbo's Enchanting    - enchant items or add sockets (it can backfire)
 *   Brentilda's Remedies  - free healing for Banjo, Kazooie and the pet
 *   Tooty's Tales         - pay her to spread tales of your deeds (fame)
 *   Captain Blubber       - fishing poles and fish
 */

FateItem gShopStock[SHOP_STOCK];

static s32 shop_depth(void) {
    s32 best = 1;
    s32 lvl;
    for (lvl = LEVEL_1_MUMBOS_MOUNTAIN; lvl <= LEVEL_A_MAD_MONSTER_MANSION; lvl++) {
        if (lvl != LEVEL_6_LAIR && itemscore_noteScores_get(lvl) > 0) {
            best = fate_maxi(best, fate_depth_for_level(lvl));
        }
    }
    return fate_maxi(best, gFate.level / 3 + 1);
}

void fate_shop_restock(void) {
    s32 depth = shop_depth();
    s32 i;
    for (i = 0; i < SHOP_STOCK; i++) {
        fate_item_generate(&gShopStock[i], depth, fate_chance(4000) ? RARITY_MAGIC : RARITY_NORMAL, 0);
        fate_identify(&gShopStock[i]);
    }
}

bool fate_shop_buy_item(s32 index) {
    s32 price;
    char name[64];
    char msg[96];

    if (index < 0 || index >= SHOP_STOCK || fate_item_empty(&gShopStock[index])) {
        return FALSE;
    }
    if (fate_inventory_free_slots() == 0) {
        fate_message("Your pack is full.", MSG_COLOR_BAD);
        return FALSE;
    }
    price = fate_item_buy_price(&gShopStock[index]);
    if (!fate_spend_gold(price)) {
        return FALSE;
    }
    fate_inventory_add(&gShopStock[index]);
    fate_item_name(&gShopStock[index], name, sizeof(name));
    msg[0] = '\0';
    fate_strcat(msg, "Bought ");
    fate_strcat(msg, name);
    fate_message(msg, fate_rarity_color(gShopStock[index].rarity));
    fate_item_clear(&gShopStock[index]);
    fate_play_jingle(COMUSIC_0_DING_A);
    return TRUE;
}

bool fate_shop_buy_consumable(s32 cons) {
    if (cons < 0 || cons >= CONS_COUNT || gFate.consumables[cons] >= 99) {
        return FALSE;
    }
    if (!fate_spend_gold(gConsumablePrices[cons])) {
        return FALSE;
    }
    gFate.consumables[cons]++;
    fate_play_jingle(COMUSIC_0_DING_A);
    return TRUE;
}

bool fate_shop_buy_spell(s32 spell) {
    char msg[96];
    if (spell < 0 || spell >= SPELL_COUNT || fate_spell_known(spell)) {
        return FALSE;
    }
    if (gDerived.stat[STAT_MAG] < gSpellDefs[spell].reqMag) {
        fate_message("Cheato: \"Your Magic is too feeble to read me, bear!\"", MSG_COLOR_BAD);
        fate_play_jingle(COMUSIC_2C_BUZZER);
        return FALSE;
    }
    if (!fate_spend_gold(gSpellDefs[spell].price)) {
        return FALSE;
    }
    gFate.spellsKnown |= 1u << spell;
    gFate.selectedSpell = (u8)spell;
    msg[0] = '\0';
    fate_strcat(msg, "You learned ");
    fate_strcat(msg, gSpellDefs[spell].name);
    fate_strcat(msg, "! Cast it with D-Pad Right.");
    fate_message(msg, MSG_COLOR_FAME);
    fate_play_jingle(COMUSIC_79_CHEATO);
    return TRUE;
}

s32 fate_pole_price(void) {
    static const u16 prices[4] = { 100, 600, 2000, 0 };
    return gFate.poleTier < 3 ? prices[gFate.poleTier] : 0;
}

bool fate_shop_buy_pole(void) {
    static const char *names[3] = { "a Fishing Pole", "a Sturdy Fishing Pole", "Blubber's Golden Rod" };
    char msg[64];
    if (gFate.poleTier >= 3) {
        return FALSE;
    }
    if (!fate_spend_gold(fate_pole_price())) {
        return FALSE;
    }
    msg[0] = '\0';
    fate_strcat(msg, "You bought ");
    fate_strcat(msg, names[gFate.poleTier]);
    fate_strcat(msg, "! Wade into water and press L.");
    gFate.poleTier++;
    fate_message(msg, MSG_COLOR_PET);
    fate_play_jingle(COMUSIC_0_DING_A);
    return TRUE;
}

bool fate_shop_buy_fish(s32 type) {
    s32 idx = type * FISH_SIZE_COUNT + FISH_SIZE_MEDIUM;
    if (type < 0 || type >= FISH_DOGFISH + 1 || type == FISH_OLD_BOOT || gFate.fish[idx] >= 255) {
        return FALSE;
    }
    if (!fate_spend_gold(fate_fish_price(type, FISH_SIZE_MEDIUM) * 3)) {
        return FALSE;
    }
    gFate.fish[idx]++;
    fate_play_jingle(COMUSIC_0_DING_A);
    return TRUE;
}

bool fate_shop_sell_inventory(s32 invIndex) {
    FateItem *it;
    s32 price;
    char msg[64];

    if (invIndex < 0 || invIndex >= INV_SIZE) {
        return FALSE;
    }
    it = &gFate.inv[invIndex];
    if (fate_item_empty(it) || it->rarity == RARITY_ARTIFACT) {
        return FALSE;
    }
    price = fate_item_sell_price(it);
    fate_add_gold(price);
    fate_item_clear(it);
    msg[0] = '\0';
    fate_strcat(msg, "Sold for ");
    fate_strcat_int(msg, price);
    fate_strcat(msg, " gold.");
    fate_message(msg, MSG_COLOR_GOLD);
    fate_play_sfx(SFX_C5_TWINKLY_POP);
    return TRUE;
}

s32 fate_shop_sell_all_fish(void) {
    s32 gold = 0;
    s32 t, sz;
    char msg[64];
    for (t = 0; t < FISH_COUNT; t++) {
        for (sz = 0; sz < FISH_SIZE_COUNT; sz++) {
            s32 idx = t * FISH_SIZE_COUNT + sz;
            gold += gFate.fish[idx] * fate_fish_price(t, sz);
            gFate.fish[idx] = 0;
        }
    }
    if (gold > 0) {
        fate_add_gold(gold);
        msg[0] = '\0';
        fate_strcat(msg, "Captain Blubber pays ");
        fate_strcat_int(msg, gold);
        fate_strcat(msg, " gold for your catch.");
        fate_message(msg, MSG_COLOR_GOLD);
    }
    return gold;
}

/* Sells every plain (white) item in the pack. */
s32 fate_shop_sell_junk(void) {
    s32 gold = 0;
    s32 i;
    char msg[64];
    for (i = 0; i < INV_SIZE; i++) {
        FateItem *it = &gFate.inv[i];
        if (!fate_item_empty(it) && it->rarity == RARITY_NORMAL && !(it->flags & ITEMF_HEIRLOOM) && fate_item_sockets(it) == 0) {
            gold += fate_item_sell_price(it);
            fate_item_clear(it);
        }
    }
    if (gold > 0) {
        fate_add_gold(gold);
        msg[0] = '\0';
        fate_strcat(msg, "Sold your junk for ");
        fate_strcat_int(msg, gold);
        fate_strcat(msg, " gold.");
        fate_message(msg, MSG_COLOR_GOLD);
    }
    return gold;
}

s32 fate_identify_cost(const FateItem *item) {
    return (20 + item->ilvl * 5) * (item->rarity >= RARITY_UNIQUE ? 2 : 1);
}

bool fate_shop_identify(s32 invIndex) {
    FateItem *it;
    if (invIndex < 0 || invIndex >= INV_SIZE) {
        return FALSE;
    }
    it = &gFate.inv[invIndex];
    if (fate_item_empty(it) || fate_item_identified(it)) {
        return FALSE;
    }
    if (!fate_spend_gold(fate_identify_cost(it))) {
        return FALSE;
    }
    fate_identify(it);
    fate_play_jingle(COMUSIC_0_DING_A);
    return TRUE;
}

/* ------------------------------------------------------------------ */
/* Mumbo's enchanting                                                  */
/* ------------------------------------------------------------------ */

s32 fate_enchant_cost(const FateItem *item) {
    return 100 + fate_item_value(item) / 2;
}

static s32 free_affix_slot(const FateItem *item) {
    s32 start = (item->rarity == RARITY_UNIQUE) ? 1 : 0;
    s32 i;
    for (i = start; i < 3; i++) {
        if (item->affix[i] == 0) {
            return i;
        }
    }
    return -1;
}

const char *fate_enchant(FateItem *item) {
    s32 r;
    s32 kind;

    if (fate_item_empty(item) || item->rarity == RARITY_ARTIFACT) {
        return "Mumbo: \"Mumbo no can enchant that!\"";
    }
    if (!fate_item_identified(item)) {
        return "Mumbo: \"Identify first! Mumbo not enchant mystery!\"";
    }
    if (!fate_spend_gold(fate_enchant_cost(item))) {
        return "Mumbo: \"Mumbo want more gold!\"";
    }
    kind = fate_item_base(item)->kind;
    r = fate_rand_range(0, 99);
    fate_play_jingle(COMUSIC_1D_MUMBO_TRANSFORMATION);
    if (r < 60) {
        s32 slot = free_affix_slot(item);
        if (slot >= 0) {
            s32 wantSuffix = (slot == 0) ? 0 : (slot == 1 ? 1 : fate_rand_range(0, 1));
            s32 id = 0;
            s32 tries;
            for (tries = 0; tries < 20 && id == 0; tries++) {
                s32 cand = fate_rand_range(1, gAffixCount - 1);
                if (gAffixes[cand].isSuffix == wantSuffix && (gAffixes[cand].kindMask & KINDBIT(kind)) && gAffixes[cand].minIlvl <= item->ilvl) {
                    id = cand;
                }
            }
            if (id == 0) {
                return "Mumbo: \"Magic fizzle. Mumbo keep gold, sorry.\"";
            }
            item->affix[slot] = (u8)id;
            item->roll[slot] = (u8)fate_rand_range(64, 255);
            if (item->rarity == RARITY_NORMAL) {
                item->rarity = RARITY_MAGIC;
            }
            item->flags |= ITEMF_IDENTIFIED;
            fate_recalc();
            return "Mumbo: \"Ooga booga! New magic on item!\"";
        } else {
            s32 i = fate_rand_range(0, 2);
            item->roll[i] = (u8)fate_mini(255, item->roll[i] + fate_rand_range(40, 120));
            fate_recalc();
            return "Mumbo: \"Mumbo make magic stronger!\"";
        }
    } else if (r < 85) {
        return "Mumbo: \"Hmm. Nothing happen. Mumbo keep gold.\"";
    } else if (r < 95) {
        if (item->rarity == RARITY_UNIQUE) {
            item->affix[1] = item->affix[2] = 0;
        } else {
            item->affix[0] = item->affix[1] = item->affix[2] = 0;
            if (item->rarity == RARITY_MAGIC) {
                item->rarity = RARITY_NORMAL;
            }
        }
        fate_recalc();
        return "Mumbo: \"Oops. All magic gone. Mumbo very sorry.\"";
    } else {
        item->flags |= ITEMF_CURSED;
        fate_recalc();
        return "Mumbo: \"Uh oh. Item now CURSED. Bad juju!\"";
    }
}

s32 fate_socket_cost(const FateItem *item) {
    return 200 + fate_item_value(item);
}

const char *fate_add_socket(FateItem *item) {
    s32 n;
    if (fate_item_empty(item) || item->rarity == RARITY_ARTIFACT) {
        return "Mumbo: \"Mumbo no can socket that!\"";
    }
    n = fate_item_sockets(item);
    if (n >= 3) {
        return "Mumbo: \"Item already full of holes!\"";
    }
    if (!fate_spend_gold(fate_socket_cost(item))) {
        return "Mumbo: \"Mumbo want more gold!\"";
    }
    fate_play_jingle(COMUSIC_1D_MUMBO_TRANSFORMATION);
    if (fate_chance(7000)) {
        item->flags = (item->flags & ~ITEMF_SOCKETS_MASK) | (((n + 1) << ITEMF_SOCKETS_SHIFT) & ITEMF_SOCKETS_MASK);
        return "Mumbo: \"Mumbo poke hole for gem. Success!\"";
    }
    return "Mumbo: \"Hole close up again. Mumbo keep gold.\"";
}

bool fate_socket_gem(FateItem *item, s32 gem) {
    s32 n = fate_item_sockets(item);
    s32 i;
    if (gem <= 0 || gem >= gGemCount || gem >= 16 || gFate.gems[gem] == 0) {
        return FALSE;
    }
    for (i = 0; i < n; i++) {
        if (item->gem[i] == 0) {
            item->gem[i] = (u8)gem;
            gFate.gems[gem]--;
            fate_recalc();
            fate_play_jingle(COMUSIC_0_DING_A);
            return TRUE;
        }
    }
    fate_message("That item has no empty sockets.", MSG_COLOR_BAD);
    return FALSE;
}

/* ------------------------------------------------------------------ */
/* Tooty and Brentilda                                                 */
/* ------------------------------------------------------------------ */

s32 fate_fame_cost(void) {
    return 200 + (s32)gFate.fameBought * 4;
}

bool fate_buy_fame(void) {
    if (!fate_spend_gold(fate_fame_cost())) {
        return FALSE;
    }
    gFate.fameBought += 25;
    fate_message("Tooty tells everyone in Grove about your adventures!", MSG_COLOR_FAME);
    fate_gain_fame(25);
    return TRUE;
}

void fate_healer(void) {
    fate_heal_full();
    fate_pet_heal_full();
    fate_message("Brentilda: \"There you go, dearie. Good as new!\"", MSG_COLOR_XP);
    fate_play_jingle(COMUSIC_7A_BRENTILDA);
}
