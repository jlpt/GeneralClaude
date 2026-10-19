#include "fate.h"

/* ------------------------------------------------------------------ */
/* Basics                                                              */
/* ------------------------------------------------------------------ */

bool fate_item_empty(const FateItem *item) {
    return item->base == 0 || item->base >= gItemBaseCount;
}

void fate_item_clear(FateItem *item) {
    bzero(item, sizeof(*item));
}

const FateItemBase *fate_item_base(const FateItem *item) {
    if (fate_item_empty(item)) {
        return &gItemBases[0];
    }
    return &gItemBases[item->base];
}

bool fate_item_identified(const FateItem *item) {
    return item->rarity == RARITY_NORMAL || item->rarity == RARITY_ARTIFACT || (item->flags & ITEMF_IDENTIFIED);
}

s32 fate_item_sockets(const FateItem *item) {
    return (item->flags & ITEMF_SOCKETS_MASK) >> ITEMF_SOCKETS_SHIFT;
}

static void set_sockets(FateItem *item, s32 n) {
    item->flags = (item->flags & ~ITEMF_SOCKETS_MASK) | ((n << ITEMF_SOCKETS_SHIFT) & ITEMF_SOCKETS_MASK);
}

u32 fate_rarity_color(s32 rarity) {
    switch (rarity) {
        case RARITY_MAGIC:     return 0x6E9BFFFF;
        case RARITY_UNIQUE:    return 0xE8C14BFF;
        case RARITY_ELITE:     return 0x5FD35FFF;
        case RARITY_LEGENDARY: return 0xFF8C2AFF;
        case RARITY_ARTIFACT:  return 0xD08CFFFF;
        default:               return 0xE8E8E8FF;
    }
}

/* ------------------------------------------------------------------ */
/* Effects                                                             */
/* ------------------------------------------------------------------ */

static s32 affix_value(const FateItem *item, s32 slot) {
    const FateAffix *a;
    s32 v;

    if (item->affix[slot] == 0 || item->affix[slot] >= gAffixCount) {
        return 0;
    }
    a = &gAffixes[item->affix[slot]];
    v = a->min + (a->max - a->min) * item->roll[slot] / 255 + item->ilvl * a->perLevel / 10;
    if (item->rarity == RARITY_ELITE) {
        v = v * 120 / 100;
    } else if (item->rarity == RARITY_LEGENDARY) {
        v = v * 140 / 100;
    }
    if (item->flags & ITEMF_HEIRLOOM) {
        v = v + v * 25 * item->enchant / 100;
    }
    if (item->flags & ITEMF_CURSED) {
        v = -v / 2;
    }
    return v < 1 && !(item->flags & ITEMF_CURSED) ? 1 : v;
}

static s32 unique_value(const FateItem *item, s32 i) {
    const FateUnique *u = &gUniques[item->affix[0]];
    s32 v = u->value[i];
    s32 over = (s32)item->ilvl - (s32)u->minDepth;
    if (over > 0) {
        v = v + v * over * 3 / 100;
    }
    if (item->flags & ITEMF_HEIRLOOM) {
        v = v + v * 25 * item->enchant / 100;
    }
    return v;
}

s32 fate_item_effect_total(const FateItem *item, s32 effect) {
    s32 total = 0;
    s32 i;

    if (fate_item_empty(item) || !fate_item_identified(item)) {
        return 0;
    }
    if (item->rarity == RARITY_UNIQUE) {
        if (item->affix[0] > 0 && item->affix[0] < gUniqueCount) {
            const FateUnique *u = &gUniques[item->affix[0]];
            for (i = 0; i < 3; i++) {
                if (u->effect[i] == effect) {
                    total += unique_value(item, i);
                }
            }
        }
        /* Enchanting a unique adds regular affixes in slots 1-2. */
        for (i = 1; i < 3; i++) {
            if (item->affix[i] && gAffixes[item->affix[i]].effect == effect) {
                total += affix_value(item, i);
            }
        }
    } else if (item->rarity != RARITY_ARTIFACT) {
        for (i = 0; i < 3; i++) {
            if (item->affix[i] && item->affix[i] < gAffixCount && gAffixes[item->affix[i]].effect == effect) {
                total += affix_value(item, i);
            }
        }
    }
    for (i = 0; i < 3; i++) {
        if (item->gem[i] > 0 && item->gem[i] < gGemCount && gGems[item->gem[i]].effect == effect) {
            total += gGems[item->gem[i]].value;
        }
    }
    return total;
}

/* ------------------------------------------------------------------ */
/* Naming and description                                              */
/* ------------------------------------------------------------------ */

static void safe_cat(char *out, s32 cap, const char *s) {
    s32 len = fate_strlen(out);
    while (*s && len < cap - 1) {
        out[len++] = *s++;
    }
    out[len] = '\0';
}

static void safe_cat_int(char *out, s32 cap, s32 v) {
    char tmp[16];
    tmp[0] = '\0';
    fate_strcat_int(tmp, v);
    safe_cat(out, cap, tmp);
}

void fate_item_name(const FateItem *item, char *out, s32 cap) {
    const FateItemBase *b = fate_item_base(item);
    s32 i;

    out[0] = '\0';
    if (fate_item_empty(item)) {
        safe_cat(out, cap, "(empty)");
        return;
    }
    if (item->rarity == RARITY_ARTIFACT) {
        safe_cat(out, cap, gArtifactNames[item->affix[0] % gArtifactNameCount]);
        return;
    }
    if (!fate_item_identified(item)) {
        safe_cat(out, cap, "Unidentified ");
        safe_cat(out, cap, b->name);
        return;
    }
    if (item->rarity == RARITY_UNIQUE && item->affix[0] > 0 && item->affix[0] < gUniqueCount) {
        safe_cat(out, cap, gUniques[item->affix[0]].name);
    } else {
        /* prefix(es) first, then the base, then the suffix */
        for (i = 0; i < 3; i++) {
            if (item->affix[i] && !gAffixes[item->affix[i]].isSuffix) {
                safe_cat(out, cap, gAffixes[item->affix[i]].name);
                safe_cat(out, cap, " ");
                break;
            }
        }
        safe_cat(out, cap, b->name);
        for (i = 0; i < 3; i++) {
            if (item->affix[i] && gAffixes[item->affix[i]].isSuffix) {
                safe_cat(out, cap, " ");
                safe_cat(out, cap, gAffixes[item->affix[i]].name);
                break;
            }
        }
    }
    if (item->flags & ITEMF_HEIRLOOM) {
        safe_cat(out, cap, " (Heirloom +");
        safe_cat_int(out, cap, item->enchant * 25);
        safe_cat(out, cap, "%)");
    }
    if (item->flags & ITEMF_CURSED) {
        safe_cat(out, cap, " (Cursed)");
    }
}

static void describe_effect(char *out, s32 cap, s32 effect, s32 value) {
    char tmp[16];
    if (effect == FX_NONE || value == 0) {
        return;
    }
    tmp[0] = '\0';
    fate_strcat_signed(tmp, value);
    safe_cat(out, cap, tmp);
    safe_cat(out, cap, gEffectFormats[effect]);
    safe_cat(out, cap, "\n");
}

void fate_item_describe(const FateItem *item, char *out, s32 cap) {
    const FateItemBase *b = fate_item_base(item);
    s32 i;
    s32 sockets;
    s32 fameReq;

    out[0] = '\0';
    if (fate_item_empty(item)) {
        return;
    }
    safe_cat(out, cap, gRarityNames[item->rarity]);
    safe_cat(out, cap, " ");
    switch (b->kind) {
        case KIND_BANJO_WEAPON:   safe_cat(out, cap, "Paw Weapon (Banjo)"); break;
        case KIND_KAZOOIE_WEAPON: safe_cat(out, cap, "Beak Weapon (Kazooie)"); break;
        case KIND_HELM:           safe_cat(out, cap, "Helm"); break;
        case KIND_ARMOR:          safe_cat(out, cap, "Armor"); break;
        case KIND_SHIELD:         safe_cat(out, cap, "Backpack Shield"); break;
        case KIND_BOOTS:          safe_cat(out, cap, "Boots"); break;
        case KIND_AMULET:         safe_cat(out, cap, "Amulet"); break;
        case KIND_RING:           safe_cat(out, cap, "Ring"); break;
        case KIND_ARTIFACT:       safe_cat(out, cap, "Quest Artifact"); break;
    }
    safe_cat(out, cap, "  (item level ");
    safe_cat_int(out, cap, item->ilvl);
    safe_cat(out, cap, ")\n");

    if (b->kind == KIND_BANJO_WEAPON || b->kind == KIND_KAZOOIE_WEAPON) {
        s32 lo = b->a, hi = b->b;
        s32 over = (s32)item->ilvl - (s32)b->tier;
        s32 pct = 100 + (over > 0 ? over * 5 : 0);
        if (item->rarity == RARITY_ELITE) pct += 50;
        if (item->rarity == RARITY_LEGENDARY) pct += 100;
        if (item->flags & ITEMF_HEIRLOOM) pct += 25 * item->enchant;
        safe_cat(out, cap, "Damage: ");
        safe_cat_int(out, cap, lo * pct / 100);
        safe_cat(out, cap, " - ");
        safe_cat_int(out, cap, hi * pct / 100);
        safe_cat(out, cap, "\n");
    } else if (b->a > 0) {
        s32 over = (s32)item->ilvl - (s32)b->tier;
        s32 pct = 100 + (over > 0 ? over * 5 : 0);
        if (item->rarity == RARITY_ELITE) pct += 50;
        if (item->rarity == RARITY_LEGENDARY) pct += 100;
        if (item->flags & ITEMF_HEIRLOOM) pct += 25 * item->enchant;
        safe_cat(out, cap, "Armor: ");
        safe_cat_int(out, cap, b->a * pct / 100);
        safe_cat(out, cap, "\n");
    }
    if (b->reqStr || b->reqDex || b->reqMag) {
        safe_cat(out, cap, "Requires");
        if (b->reqStr) { safe_cat(out, cap, " STR "); safe_cat_int(out, cap, b->reqStr); }
        if (b->reqDex) { safe_cat(out, cap, " DEX "); safe_cat_int(out, cap, b->reqDex); }
        if (b->reqMag) { safe_cat(out, cap, " MAG "); safe_cat_int(out, cap, b->reqMag); }
        safe_cat(out, cap, "\n");
    }
    fameReq = fate_item_fame_req(item);
    if (fameReq > 0) {
        safe_cat(out, cap, "Requires Fame level ");
        safe_cat_int(out, cap, fameReq);
        safe_cat(out, cap, "\n");
    }

    if (!fate_item_identified(item)) {
        safe_cat(out, cap, "Its enchantments are unknown. Identify it with a Scroll of Identify,\nthe Discern spell, or at Bottles' Burrow.\n");
    } else if (item->rarity == RARITY_ARTIFACT) {
        safe_cat(out, cap, "Return it to whoever asked for it in Grove.\n");
    } else {
        for (i = 0; i < FX_COUNT; i++) {
            describe_effect(out, cap, i, fate_item_effect_total(item, i));
        }
    }

    sockets = fate_item_sockets(item);
    if (sockets > 0) {
        safe_cat(out, cap, "Sockets:");
        for (i = 0; i < sockets; i++) {
            safe_cat(out, cap, " [");
            safe_cat(out, cap, item->gem[i] ? gGems[item->gem[i]].name : "empty");
            safe_cat(out, cap, "]");
        }
        safe_cat(out, cap, "\n");
    }
    safe_cat(out, cap, "Sells for ");
    safe_cat_int(out, cap, fate_item_sell_price(item));
    safe_cat(out, cap, " gold");
}

/* ------------------------------------------------------------------ */
/* Value and requirements                                              */
/* ------------------------------------------------------------------ */

s32 fate_item_value(const FateItem *item) {
    const FateItemBase *b = fate_item_base(item);
    s32 v;
    s32 i;
    static const u8 rarityMult[RARITY_COUNT] = { 10, 25, 80, 60, 150, 10 };

    if (fate_item_empty(item)) {
        return 0;
    }
    v = b->price * rarityMult[item->rarity] / 10;
    v = v + v * item->ilvl * 8 / 100;
    for (i = 0; i < 3; i++) {
        if (item->affix[i]) {
            v += 10 + item->ilvl * 4;
        }
        if (item->gem[i] && item->gem[i] < gGemCount) {
            v += gGems[item->gem[i]].price;
        }
    }
    v += fate_item_sockets(item) * 50;
    return v;
}

s32 fate_item_sell_price(const FateItem *item) {
    s32 v = fate_item_value(item) / 4;
    if (!fate_item_identified(item)) {
        v /= 2;
    }
    if (item->flags & ITEMF_HEIRLOOM) {
        v = 1;
    }
    return v < 1 ? 1 : v;
}

s32 fate_item_buy_price(const FateItem *item) {
    return fate_item_value(item);
}

s32 fate_item_fame_req(const FateItem *item) {
    if (item->rarity == RARITY_ELITE) {
        return 3 + item->ilvl / 10;
    }
    if (item->rarity == RARITY_LEGENDARY) {
        return 6 + item->ilvl / 10;
    }
    return 0;
}

bool fate_item_can_equip(const FateItem *item, char *reason, s32 cap) {
    const FateItemBase *b = fate_item_base(item);

    if (reason) {
        reason[0] = '\0';
    }
    if (fate_item_empty(item) || b->kind == KIND_ARTIFACT) {
        if (reason) safe_cat(reason, cap, "That can't be equipped.");
        return FALSE;
    }
    if (!fate_item_identified(item)) {
        if (reason) safe_cat(reason, cap, "Identify it first.");
        return FALSE;
    }
    if (gDerived.stat[STAT_STR] < b->reqStr) {
        if (reason) safe_cat(reason, cap, "Not enough Strength.");
        return FALSE;
    }
    if (gDerived.stat[STAT_DEX] < b->reqDex) {
        if (reason) safe_cat(reason, cap, "Not enough Dexterity.");
        return FALSE;
    }
    if (gDerived.stat[STAT_MAG] < b->reqMag) {
        if (reason) safe_cat(reason, cap, "Not enough Magic.");
        return FALSE;
    }
    if (fate_fame_level() < fate_item_fame_req(item)) {
        if (reason) safe_cat(reason, cap, "You aren't famous enough to wield it.");
        return FALSE;
    }
    return TRUE;
}

s32 fate_slot_for_kind(s32 kind, s32 preferred) {
    switch (kind) {
        case KIND_BANJO_WEAPON:   return SLOT_BANJO_WEAPON;
        case KIND_KAZOOIE_WEAPON: return SLOT_KAZOOIE_WEAPON;
        case KIND_HELM:           return SLOT_HELM;
        case KIND_ARMOR:          return SLOT_ARMOR;
        case KIND_SHIELD:         return SLOT_SHIELD;
        case KIND_BOOTS:          return SLOT_BOOTS;
        case KIND_AMULET:         return SLOT_AMULET;
        case KIND_RING:
            if (preferred == SLOT_RING1 || preferred == SLOT_RING2) {
                return preferred;
            }
            if (fate_item_empty(&gFate.equip[SLOT_RING1])) {
                return SLOT_RING1;
            }
            return fate_item_empty(&gFate.equip[SLOT_RING2]) ? SLOT_RING2 : SLOT_RING1;
        default:
            return -1;
    }
}

/* ------------------------------------------------------------------ */
/* Generation                                                          */
/* ------------------------------------------------------------------ */

static s32 pick_affix(s32 kind, s32 ilvl, s32 wantSuffix, s32 avoidEffect) {
    s32 candidates[64];
    s32 n = 0;
    s32 i;

    for (i = 1; i < gAffixCount && n < 64; i++) {
        const FateAffix *a = &gAffixes[i];
        if (a->isSuffix != wantSuffix) continue;
        if (!(a->kindMask & KINDBIT(kind))) continue;
        if (a->minIlvl > ilvl) continue;
        if (a->effect == avoidEffect) continue;
        candidates[n++] = i;
    }
    if (n == 0) {
        return 0;
    }
    return candidates[fate_rand_range(0, n - 1)];
}

static s32 pick_base(s32 depth) {
    s32 candidates[96];
    s32 n = 0;
    s32 i;
    s32 tierCap = fate_clampi(depth, 1, 10);
    s32 tierFloor = tierCap - 3;

    for (i = 1; i < gItemBaseCount && n < 96; i++) {
        const FateItemBase *b = &gItemBases[i];
        if (b->kind == KIND_NONE || b->kind == KIND_ARTIFACT) continue;
        if (b->tier > tierCap || b->tier < tierFloor) continue;
        candidates[n++] = i;
    }
    if (n == 0) {
        return 1;
    }
    return candidates[fate_rand_range(0, n - 1)];
}

void fate_item_generate_base(FateItem *out, s32 base, s32 rarity, s32 ilvl) {
    const FateItemBase *b;
    s32 kind;
    s32 used;

    fate_item_clear(out);
    out->base = base;
    out->rarity = rarity;
    out->ilvl = fate_clampi(ilvl, 1, 99);
    b = fate_item_base(out);
    kind = b->kind;

    switch (rarity) {
        case RARITY_NORMAL:
            out->flags |= ITEMF_IDENTIFIED;
            if (fate_chance(1500)) {
                set_sockets(out, fate_rand_range(1, 2));
            }
            break;
        case RARITY_MAGIC:
            if (fate_chance(5000)) {
                out->affix[0] = pick_affix(kind, out->ilvl, 0, FX_NONE);
                out->affix[1] = pick_affix(kind, out->ilvl, 1, out->affix[0] ? gAffixes[out->affix[0]].effect : FX_NONE);
            } else if (fate_chance(5000)) {
                out->affix[0] = pick_affix(kind, out->ilvl, 0, FX_NONE);
            } else {
                out->affix[0] = pick_affix(kind, out->ilvl, 1, FX_NONE);
            }
            if (fate_chance(1000)) {
                set_sockets(out, 1);
            }
            break;
        case RARITY_ELITE:
            out->affix[0] = pick_affix(kind, out->ilvl, 0, FX_NONE);
            out->affix[1] = pick_affix(kind, out->ilvl, 1, out->affix[0] ? gAffixes[out->affix[0]].effect : FX_NONE);
            if (fate_chance(3000)) {
                set_sockets(out, fate_rand_range(1, 2));
            }
            break;
        case RARITY_LEGENDARY:
            out->affix[0] = pick_affix(kind, out->ilvl, 0, FX_NONE);
            used = out->affix[0] ? gAffixes[out->affix[0]].effect : FX_NONE;
            out->affix[1] = pick_affix(kind, out->ilvl, 1, used);
            out->affix[2] = pick_affix(kind, out->ilvl, fate_rand_range(0, 1),
                                       out->affix[1] ? gAffixes[out->affix[1]].effect : used);
            set_sockets(out, fate_rand_range(1, 3));
            break;
        default:
            break;
    }
    out->roll[0] = (u8)fate_rand_range(0, 255);
    out->roll[1] = (u8)fate_rand_range(0, 255);
    out->roll[2] = (u8)fate_rand_range(0, 255);

    /* A magic roll that found no valid affix is just a normal item. */
    if ((rarity == RARITY_MAGIC || rarity == RARITY_ELITE || rarity == RARITY_LEGENDARY)
        && !out->affix[0] && !out->affix[1] && !out->affix[2]) {
        if (rarity == RARITY_MAGIC) {
            out->rarity = RARITY_NORMAL;
            out->flags |= ITEMF_IDENTIFIED;
        }
    }
}

bool fate_item_generate(FateItem *out, s32 depth, s32 forceRarity, s32 magicFind) {
    s32 rarity = forceRarity;
    s32 ilvl = fate_maxi(1, depth + fate_rand_range(0, 1));
    s32 mf = 100 + magicFind;
    s32 fame = fate_fame_level();

    if (rarity < 0) {
        rarity = RARITY_NORMAL;
        if (fame >= 6 && fate_chance(150 * mf / 100)) {
            rarity = RARITY_LEGENDARY;
        } else if (fate_chance(300 * mf / 100)) {
            rarity = RARITY_UNIQUE;
        } else if (fame >= 3 && fate_chance(400 * mf / 100)) {
            rarity = RARITY_ELITE;
        } else if (fate_chance(3000 * mf / 100)) {
            rarity = RARITY_MAGIC;
        }
    }

    if (rarity == RARITY_UNIQUE) {
        s32 candidates[32];
        s32 n = 0;
        s32 i;
        for (i = 1; i < gUniqueCount && n < 32; i++) {
            if (gUniques[i].minDepth <= depth) {
                candidates[n++] = i;
            }
        }
        if (n > 0) {
            s32 u = candidates[fate_rand_range(0, n - 1)];
            fate_item_clear(out);
            out->base = gUniques[u].base;
            out->rarity = RARITY_UNIQUE;
            out->ilvl = fate_clampi(ilvl, 1, 99);
            out->affix[0] = u;
            if (fate_chance(2000)) {
                set_sockets(out, 1);
            }
            return TRUE;
        }
        rarity = RARITY_MAGIC;
    }

    fate_item_generate_base(out, pick_base(depth), rarity, ilvl);
    return TRUE;
}

/* ------------------------------------------------------------------ */
/* Inventory                                                           */
/* ------------------------------------------------------------------ */

bool fate_inventory_add(const FateItem *item) {
    s32 i;
    for (i = 0; i < INV_SIZE; i++) {
        if (fate_item_empty(&gFate.inv[i])) {
            gFate.inv[i] = *item;
            return TRUE;
        }
    }
    return FALSE;
}

bool fate_pet_bag_add(const FateItem *item) {
    s32 i;
    for (i = 0; i < PET_BAG_SIZE; i++) {
        if (fate_item_empty(&gFate.petBag[i])) {
            gFate.petBag[i] = *item;
            return TRUE;
        }
    }
    return FALSE;
}

s32 fate_inventory_free_slots(void) {
    s32 n = 0;
    s32 i;
    for (i = 0; i < INV_SIZE; i++) {
        if (fate_item_empty(&gFate.inv[i])) {
            n++;
        }
    }
    return n;
}

bool fate_equip_from_inventory(s32 invIndex) {
    FateItem tmp;
    char reason[64];
    s32 slot;

    if (invIndex < 0 || invIndex >= INV_SIZE || fate_item_empty(&gFate.inv[invIndex])) {
        return FALSE;
    }
    if (!fate_item_can_equip(&gFate.inv[invIndex], reason, sizeof(reason))) {
        fate_message(reason, MSG_COLOR_BAD);
        fate_play_jingle(COMUSIC_2C_BUZZER);
        return FALSE;
    }
    slot = fate_slot_for_kind(fate_item_base(&gFate.inv[invIndex])->kind, -1);
    if (slot < 0) {
        return FALSE;
    }
    tmp = gFate.equip[slot];
    gFate.equip[slot] = gFate.inv[invIndex];
    gFate.inv[invIndex] = tmp;
    fate_recalc();
    fate_play_sfx(SFX_3EB_UNKNOWN);
    return TRUE;
}

bool fate_unequip(s32 slot) {
    if (slot < 0 || slot >= SLOT_COUNT || fate_item_empty(&gFate.equip[slot])) {
        return FALSE;
    }
    if (!fate_inventory_add(&gFate.equip[slot])) {
        fate_message("Your pack is full.", MSG_COLOR_BAD);
        return FALSE;
    }
    fate_item_clear(&gFate.equip[slot]);
    fate_recalc();
    return TRUE;
}

void fate_identify(FateItem *item) {
    if (!fate_item_empty(item)) {
        item->flags |= ITEMF_IDENTIFIED;
    }
}

s32 fate_identify_all(void) {
    s32 n = 0;
    s32 i;
    for (i = 0; i < INV_SIZE; i++) {
        if (!fate_item_empty(&gFate.inv[i]) && !fate_item_identified(&gFate.inv[i])) {
            fate_identify(&gFate.inv[i]);
            n++;
        }
    }
    for (i = 0; i < PET_BAG_SIZE; i++) {
        if (!fate_item_empty(&gFate.petBag[i]) && !fate_item_identified(&gFate.petBag[i])) {
            fate_identify(&gFate.petBag[i]);
            n++;
        }
    }
    return n;
}
