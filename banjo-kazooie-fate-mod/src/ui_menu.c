#include "ui_common.h"

/*
 * The Fate menu (D-Pad Up). Opening it freezes the world. Tabs:
 *   Character  attributes, skills, fame, retirement
 *   Inventory  equipment, pack, consumables, gems
 *   Pet        pet stats, fish feeding, pet bag, send to Grove
 *   Spells     known spells and the active spell
 *   Quests     quest log (and the quest board while in Grove)
 *   Grove      the shops (only in Spiral Mountain)
 * Gruntilda's bargain (Fate's death choice) uses the same context.
 *
 * Buttons carry an action code and argument in their callback userdata.
 * Callbacks only change state and request a rebuild; the rebuild itself
 * happens next frame in fate_menu_frame.
 */

enum menu_action {
    ACT_NONE,
    ACT_TAB,
    ACT_CLOSE,
    ACT_RAISE_STAT,
    ACT_RAISE_SKILL,
    ACT_SELECT,
    ACT_EQUIP,
    ACT_UNEQUIP,
    ACT_GIVE_PET,
    ACT_DROP,
    ACT_SELL,
    ACT_IDENTIFY_SCROLL,
    ACT_SOCKET_GEM,
    ACT_USE_RETURN,
    ACT_PET_EQUIP,
    ACT_PET_UNEQUIP,
    ACT_PET_TAKE,
    ACT_PET_SEND,
    ACT_FEED_FISH,
    ACT_PET_POTION,
    ACT_SELECT_SPELL,
    ACT_QUEST_ACCEPT,
    ACT_QUEST_TURNIN,
    ACT_QUEST_ABANDON,
    ACT_GROVE_SECTION,
    ACT_BUY_CONS,
    ACT_BUY_ITEM,
    ACT_BUY_SPELL,
    ACT_IDENTIFY_PAID,
    ACT_SELL_JUNK,
    ACT_ENCHANT,
    ACT_SOCKET_ADD,
    ACT_HEAL,
    ACT_BUY_FAME,
    ACT_BUY_POLE,
    ACT_BUY_FISH,
    ACT_SELL_FISH,
    ACT_PORTAL,
    ACT_RETIRE_MODE,
    ACT_RETIRE_ITEM,
    ACT_NIGHTMARE,
    ACT_BARGAIN,
    ACT_PAGE,
};

enum sel_kind {
    SEL_NONE,
    SEL_INV,
    SEL_EQUIP,
    SEL_SHOP,
    SEL_PETBAG,
    SEL_PETEQUIP,
};

enum grove_section {
    GROVE_BOTTLES,
    GROVE_GOBI,
    GROVE_CHEATO,
    GROVE_MUMBO,
    GROVE_BRENTILDA,
    GROVE_TOOTY,
    GROVE_BLUBBER,
    GROVE_COUNT
};

static const char *sGroveNames[GROVE_COUNT] = {
    "Bottles' Burrow",
    "Gobi's Arms and Armor",
    "Cheato's Spellbooks",
    "Mumbo's Enchanting",
    "Brentilda's Remedies",
    "Tooty's Tales",
    "Captain Blubber's Fish",
};

static const char *sTabNames[TAB_COUNT] = { "Character", "Inventory", "Pet", "Spells", "Quests", "Grove" };

static RecompuiContext sCtx = RECOMPUI_NULL_CONTEXT;
static RecompuiResource sRoot;
static RecompuiResource sPanel;
static RecompuiResource sBody;
static bool sShown;
static bool sRebuild;
static bool sRetireMode;
static s32 sTab;
static s32 sGrove;
static s32 sSelKind;
static s32 sSelIndex;
static s32 sPage;
static s32 sWorldWasEnabled;
static char sNotice[128];

/* ------------------------------------------------------------------ */
/* Widgets                                                             */
/* ------------------------------------------------------------------ */

static void on_click(RecompuiResource resource, const RecompuiEventData *event, void *userdata);

static RecompuiResource button(RecompuiResource parent, const char *text, s32 action, s32 arg, bool primary) {
    RecompuiResource b = recompui_create_button(sCtx, parent, text, primary ? BUTTONSTYLE_PRIMARY : BUTTONSTYLE_SECONDARY);
    recompui_set_font_size(b, 18.0f, UNIT_DP);
    recompui_set_padding(b, 6.0f, UNIT_DP);
    recompui_register_callback(b, on_click, (void *)(u32)(((action & 0xFFFF) << 16) | (arg & 0xFFFF)));
    return b;
}

static RecompuiResource col(RecompuiResource parent, f32 width) {
    RecompuiResource c = ui_box(sCtx, parent, FALSE);
    recompui_set_gap(c, 6.0f, UNIT_DP);
    if (width > 0.0f) {
        recompui_set_width(c, width, UNIT_DP);
    } else {
        recompui_set_flex_grow(c, 1.0f);
    }
    return c;
}

static RecompuiResource row(RecompuiResource parent) {
    RecompuiResource r = ui_box(sCtx, parent, TRUE);
    recompui_set_gap(r, 8.0f, UNIT_DP);
    recompui_set_align_items(r, ALIGN_ITEMS_CENTER);
    return r;
}

static RecompuiResource inset(RecompuiResource parent, f32 width) {
    RecompuiResource c = col(parent, width);
    ui_set_bg(c, UI_INSET_BG);
    ui_set_border(c, 0x6B5530FF, 1.0f, 8.0f);
    recompui_set_padding(c, 10.0f, UNIT_DP);
    return c;
}

static RecompuiResource scroller(RecompuiResource parent, f32 width, f32 height) {
    RecompuiResource c = inset(parent, width);
    recompui_set_max_height(c, height, UNIT_DP);
    recompui_set_overflow_y(c, OVERFLOW_AUTO);
    return c;
}

static RecompuiResource text(RecompuiResource parent, const char *t, f32 size, u32 color) {
    return ui_label(sCtx, parent, t, size, color);
}

static RecompuiResource header(RecompuiResource parent, const char *t) {
    RecompuiResource h = text(parent, t, 24.0f, UI_HEADER);
    recompui_set_font_weight(h, 700);
    return h;
}

static void notice(const char *msg) {
    s32 i = 0;
    while (msg[i] && i < 127) {
        sNotice[i] = msg[i];
        i++;
    }
    sNotice[i] = '\0';
}

static void stat_line(RecompuiResource parent, const char *label, s32 value, const char *suffix) {
    char buf[96];
    buf[0] = '\0';
    fate_strcat(buf, label);
    fate_strcat(buf, ": ");
    fate_strcat_int(buf, value);
    fate_strcat(buf, suffix);
    text(parent, buf, 18.0f, UI_TEXT);
}

static FateItem *selected_item(void) {
    switch (sSelKind) {
        case SEL_INV:     return (sSelIndex >= 0 && sSelIndex < INV_SIZE) ? &gFate.inv[sSelIndex] : NULL;
        case SEL_EQUIP:   return (sSelIndex >= 0 && sSelIndex < SLOT_COUNT) ? &gFate.equip[sSelIndex] : NULL;
        case SEL_SHOP:    return (sSelIndex >= 0 && sSelIndex < SHOP_STOCK) ? &gShopStock[sSelIndex] : NULL;
        case SEL_PETBAG:  return (sSelIndex >= 0 && sSelIndex < PET_BAG_SIZE) ? &gFate.petBag[sSelIndex] : NULL;
        case SEL_PETEQUIP:return (sSelIndex >= 0 && sSelIndex < PET_EQUIP_COUNT) ? &gFate.petEquip[sSelIndex] : NULL;
        default:          return NULL;
    }
}

static void item_button(RecompuiResource parent, const FateItem *item, s32 selKind, s32 index, const char *prefix) {
    char name[96];
    char line[128];
    RecompuiResource b;
    fate_item_name(item, name, sizeof(name));
    line[0] = '\0';
    if (prefix) {
        fate_strcat(line, prefix);
    }
    fate_strcat(line, name);
    b = button(parent, line, ACT_SELECT, (selKind << 8) | index, sSelKind == selKind && sSelIndex == index);
    ui_set_text_color(b, fate_rarity_color(item->rarity));
}

static void item_details(RecompuiResource parent, const FateItem *item) {
    char name[96];
    static char desc[768];
    RecompuiResource n;
    if (item == NULL || fate_item_empty(item)) {
        text(parent, "Select an item to see its details.", 18.0f, UI_TEXT_DIM);
        return;
    }
    fate_item_name(item, name, sizeof(name));
    n = text(parent, name, 22.0f, fate_rarity_color(item->rarity));
    recompui_set_font_weight(n, 700);
    fate_item_describe(item, desc, sizeof(desc));
    text(parent, desc, 17.0f, UI_TEXT);
}

/* ------------------------------------------------------------------ */
/* Character tab                                                       */
/* ------------------------------------------------------------------ */

static void build_character(RecompuiResource body) {
    RecompuiResource c1 = inset(body, 430.0f);
    RecompuiResource c2 = inset(body, 430.0f);
    RecompuiResource c3 = inset(body, 0.0f);
    char buf[128];
    s32 i;
    u32 nextFame;

    header(c1, "Attributes");
    buf[0] = '\0';
    fate_strcat(buf, "Unspent points: ");
    fate_strcat_int(buf, gFate.statPoints);
    text(c1, buf, 18.0f, gFate.statPoints ? MSG_COLOR_XP : UI_TEXT_DIM);
    for (i = 0; i < STAT_COUNT; i++) {
        RecompuiResource r = row(c1);
        buf[0] = '\0';
        fate_strcat(buf, gStatNames[i]);
        fate_strcat(buf, "  ");
        fate_strcat_int(buf, gDerived.stat[i]);
        if (gDerived.stat[i] != gFate.stats[i]) {
            fate_strcat(buf, " (base ");
            fate_strcat_int(buf, gFate.stats[i]);
            fate_strcat(buf, ")");
        }
        text(r, buf, 19.0f, UI_TEXT);
        if (gFate.statPoints > 0) {
            button(r, "+", ACT_RAISE_STAT, i, TRUE);
        }
    }
    header(c1, "Combat");
    stat_line(c1, "Health", gDerived.maxHp, "");
    stat_line(c1, "Mana", gDerived.maxMana, "");
    stat_line(c1, "Armor", gDerived.armor, "");
    buf[0] = '\0';
    fate_strcat(buf, "Paw damage: ");
    fate_strcat_int(buf, gDerived.pawMin + gDerived.stat[STAT_STR] / 4);
    fate_strcat(buf, "-");
    fate_strcat_int(buf, gDerived.pawMax + gDerived.stat[STAT_STR] / 4);
    fate_strcat(buf, "   Beak: ");
    fate_strcat_int(buf, gDerived.beakMin);
    fate_strcat(buf, "-");
    fate_strcat_int(buf, gDerived.beakMax);
    text(c1, buf, 18.0f, UI_TEXT);
    stat_line(c1, "Critical chance", gDerived.critChance / 100, "%");
    stat_line(c1, "Block", gDerived.block, "%");
    stat_line(c1, "Dodge", gDerived.dodge, "%");
    if (gDerived.goldFind) stat_line(c1, "Gold find", gDerived.goldFind, "%");
    if (gDerived.magicFind) stat_line(c1, "Magic find", gDerived.magicFind, "%");
    if (gDerived.lifeSteal) stat_line(c1, "Life steal", gDerived.lifeSteal, "%");

    header(c2, "Skills");
    buf[0] = '\0';
    fate_strcat(buf, "Unspent points: ");
    fate_strcat_int(buf, gFate.skillPoints);
    text(c2, buf, 18.0f, gFate.skillPoints ? MSG_COLOR_XP : UI_TEXT_DIM);
    for (i = 0; i < SKILL_COUNT; i++) {
        RecompuiResource r = row(c2);
        buf[0] = '\0';
        fate_strcat(buf, gSkillNames[i]);
        fate_strcat(buf, "  ");
        fate_strcat_int(buf, gFate.skills[i]);
        text(r, buf, 18.0f, gFate.skills[i] ? UI_TEXT : UI_TEXT_DIM);
        if (gFate.skillPoints > 0 && gFate.skills[i] < FATE_SKILL_MAX) {
            button(r, "+", ACT_RAISE_SKILL, i, TRUE);
        }
    }

    header(c3, "Banjo and Kazooie");
    buf[0] = '\0';
    fate_strcat(buf, "Level ");
    fate_strcat_int(buf, gFate.level);
    fate_strcat(buf, "   Experience ");
    fate_strcat_int(buf, (s32)gFate.xp);
    fate_strcat(buf, " / ");
    fate_strcat_int(buf, (s32)fate_xp_for_level(gFate.level + 1));
    text(c3, buf, 18.0f, UI_TEXT);
    buf[0] = '\0';
    fate_strcat(buf, "Fame: ");
    fate_strcat(buf, fate_fame_title());
    fate_strcat(buf, " (level ");
    fate_strcat_int(buf, fate_fame_level());
    fate_strcat(buf, ")");
    text(c3, buf, 19.0f, MSG_COLOR_FAME);
    nextFame = fate_fame_for_level(fate_fame_level() + 1);
    buf[0] = '\0';
    fate_strcat_int(buf, (s32)gFate.fame);
    fate_strcat(buf, " / ");
    fate_strcat_int(buf, (s32)nextFame);
    fate_strcat(buf, " fame to the next title (+4 skill points)");
    text(c3, buf, 16.0f, UI_TEXT_DIM);
    stat_line(c3, "Gold", (s32)gFate.gold, "");
    stat_line(c3, "Monsters slain", (s32)gFate.kills, "");
    stat_line(c3, "Champions slain", (s32)gFate.championKills, "");
    stat_line(c3, "Quests completed", gFate.questsCompleted, "");
    stat_line(c3, "Deaths", gFate.deaths, "");
    if (gFate.generation > 0) {
        buf[0] = '\0';
        fate_strcat(buf, "Generation ");
        fate_strcat_int(buf, gFate.generation + 1);
        fate_strcat(buf, " of your line");
        text(c3, buf, 18.0f, MSG_COLOR_GOLD);
    }
    if (gFate.nightmare > 0) {
        stat_line(c3, "Nightmare tier", gFate.nightmare, " (+10 depth each)");
    }

    if (fileProgressFlag_get(FILEPROG_FC_DEFEAT_GRUNTY)) {
        header(c3, "Gruntilda is defeated!");
        if (fate_can_retire()) {
            text(c3, "Retire this hero and pass one equipped item to your heir.\nThe heirloom grows 25% stronger each generation.", 16.0f, UI_TEXT_DIM);
            button(c3, sRetireMode ? "Choose the heirloom below..." : "Retire", ACT_RETIRE_MODE, 0, TRUE);
            if (sRetireMode) {
                for (i = 0; i < SLOT_COUNT; i++) {
                    if (!fate_item_empty(&gFate.equip[i])) {
                        char name[96];
                        fate_item_name(&gFate.equip[i], name, sizeof(name));
                        button(c3, name, ACT_RETIRE_ITEM, i, FALSE);
                    }
                }
            }
        } else if (gFate.retired) {
            text(c3, "This hero has retired. Start a new game file to play as the heir.", 16.0f, UI_TEXT_DIM);
        }
        if (gFate.nightmare < 5) {
            button(c3, "Descend into Nightmare (every world +10 depth)", ACT_NIGHTMARE, 0, FALSE);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Inventory tab                                                       */
/* ------------------------------------------------------------------ */

static void build_inventory(RecompuiResource body) {
    RecompuiResource cEquip = inset(body, 470.0f);
    RecompuiResource cPack = scroller(body, 470.0f, 700.0f);
    RecompuiResource cInfo = inset(body, 0.0f);
    FateItem *sel;
    char buf[128];
    s32 i;

    header(cEquip, "Equipped");
    for (i = 0; i < SLOT_COUNT; i++) {
        char prefix[40];
        prefix[0] = '\0';
        fate_strcat(prefix, gSlotNames[i]);
        fate_strcat(prefix, ": ");
        if (fate_item_empty(&gFate.equip[i])) {
            fate_strcat(prefix, "-");
            text(cEquip, prefix, 17.0f, UI_TEXT_DIM);
        } else {
            item_button(cEquip, &gFate.equip[i], SEL_EQUIP, i, prefix);
        }
    }
    header(cEquip, "Belt");
    for (i = 0; i < CONS_COUNT; i++) {
        buf[0] = '\0';
        fate_strcat(buf, gConsumableNames[i]);
        fate_strcat(buf, " x");
        fate_strcat_int(buf, gFate.consumables[i]);
        text(cEquip, buf, 17.0f, gFate.consumables[i] ? UI_TEXT : UI_TEXT_DIM);
    }
    if (gFate.consumables[CONS_TOWN_PORTAL] > 0 && !fate_in_town()) {
        button(cEquip, "Read a Scroll of Return (go to Grove)", ACT_USE_RETURN, 0, TRUE);
    }
    {
        bool anyGem = FALSE;
        for (i = 1; i < gGemCount && i < 16; i++) {
            if (gFate.gems[i]) {
                if (!anyGem) {
                    header(cEquip, "Gems");
                    anyGem = TRUE;
                }
                buf[0] = '\0';
                fate_strcat(buf, gGems[i].name);
                fate_strcat(buf, " x");
                fate_strcat_int(buf, gFate.gems[i]);
                text(cEquip, buf, 17.0f, 0x7FFFD4FF);
            }
        }
    }

    buf[0] = '\0';
    fate_strcat(buf, "Pack (");
    fate_strcat_int(buf, INV_SIZE - fate_inventory_free_slots());
    fate_strcat(buf, "/");
    fate_strcat_int(buf, INV_SIZE);
    fate_strcat(buf, ")");
    header(cPack, buf);
    for (i = 0; i < INV_SIZE; i++) {
        if (!fate_item_empty(&gFate.inv[i])) {
            item_button(cPack, &gFate.inv[i], SEL_INV, i, NULL);
        }
    }
    if (fate_inventory_free_slots() == INV_SIZE) {
        text(cPack, "Your pack is empty. Slay monsters to find loot!", 17.0f, UI_TEXT_DIM);
    }

    header(cInfo, "Details");
    sel = selected_item();
    if (sSelKind != SEL_INV && sSelKind != SEL_EQUIP) {
        sel = NULL;
    }
    item_details(cInfo, sel);
    if (sel != NULL && !fate_item_empty(sel)) {
        RecompuiResource actions = row(cInfo);
        s32 kind = fate_item_base(sel)->kind;
        if (sSelKind == SEL_INV) {
            if (kind != KIND_ARTIFACT) {
                button(actions, "Equip", ACT_EQUIP, sSelIndex, TRUE);
            }
            if (kind == KIND_AMULET || kind == KIND_RING) {
                button(actions, "Put on pet", ACT_PET_EQUIP, sSelIndex, FALSE);
            }
            button(actions, "Give to pet", ACT_GIVE_PET, sSelIndex, FALSE);
            if (fate_in_town()) {
                buf[0] = '\0';
                fate_strcat(buf, "Sell (");
                fate_strcat_int(buf, fate_item_sell_price(sel));
                fate_strcat(buf, "g)");
                button(actions, buf, ACT_SELL, sSelIndex, FALSE);
            }
            button(actions, "Drop", ACT_DROP, sSelIndex, FALSE);
            if (!fate_item_identified(sel) && gFate.consumables[CONS_IDENTIFY] > 0) {
                button(cInfo, "Read a Scroll of Identify", ACT_IDENTIFY_SCROLL, sSelIndex, TRUE);
            }
        } else {
            button(actions, "Unequip", ACT_UNEQUIP, sSelIndex, TRUE);
        }
        if (fate_item_sockets(sel) > 0) {
            for (i = 1; i < gGemCount && i < 16; i++) {
                if (gFate.gems[i]) {
                    buf[0] = '\0';
                    fate_strcat(buf, "Socket ");
                    fate_strcat(buf, gGems[i].name);
                    button(cInfo, buf, ACT_SOCKET_GEM, i, FALSE);
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Pet tab                                                             */
/* ------------------------------------------------------------------ */

static void build_pet(RecompuiResource body) {
    RecompuiResource c1 = inset(body, 480.0f);
    RecompuiResource c2 = scroller(body, 480.0f, 700.0f);
    RecompuiResource c3 = inset(body, 0.0f);
    char buf[128];
    s32 i, t, sz;
    bool anyFish = FALSE;
    s32 bagCount = 0;

    if (gFate.petSpecies == PET_SPECIES_NONE) {
        header(c1, "No pet");
        text(c1, "Your pet is disabled in the mod settings.", 18.0f, UI_TEXT_DIM);
        return;
    }
    header(c1, fate_pet_name());
    stat_line(c1, "Level", gFate.petLevel, "");
    buf[0] = '\0';
    fate_strcat(buf, "Health: ");
    fate_strcat_int(buf, gFate.petHp);
    fate_strcat(buf, " / ");
    fate_strcat_int(buf, fate_pet_max_hp());
    text(c1, buf, 18.0f, UI_TEXT);
    stat_line(c1, "Damage per bite", fate_pet_damage(), "");
    text(c1, fate_pet_status(), 18.0f, MSG_COLOR_PET);
    if (gFate.petFormSeconds > 0) {
        stat_line(c1, "Transformation lasts", gFate.petFormSeconds, " more seconds");
    }
    for (i = 0; i < PET_BAG_SIZE; i++) {
        if (!fate_item_empty(&gFate.petBag[i])) {
            bagCount++;
        }
    }
    if (fate_pet_town_seconds() <= 0.0f) {
        buf[0] = '\0';
        fate_strcat(buf, "Send to Grove to sell loot (");
        fate_strcat_int(buf, bagCount);
        fate_strcat(buf, " items)");
        button(c1, buf, ACT_PET_SEND, 0, TRUE);
        if (gFate.consumables[CONS_HP_POTION] > 0 && gFate.petHp < fate_pet_max_hp()) {
            button(c1, "Give the pet a Health Potion", ACT_PET_POTION, 0, FALSE);
        }
    }
    header(c1, "Pet jewelry");
    for (i = 0; i < PET_EQUIP_COUNT; i++) {
        if (fate_item_empty(&gFate.petEquip[i])) {
            text(c1, i == 0 ? "Collar: -" : "Charm: -", 17.0f, UI_TEXT_DIM);
        } else {
            item_button(c1, &gFate.petEquip[i], SEL_PETEQUIP, i, i == 0 ? "Collar: " : "Charm: ");
        }
    }
    text(c1, "Put amulets and rings on your pet from the Inventory tab.", 15.0f, UI_TEXT_DIM);

    header(c2, "Fish (feed to transform)");
    for (t = 0; t < FISH_COUNT; t++) {
        for (sz = 0; sz < FISH_SIZE_COUNT; sz++) {
            s32 n = gFate.fish[t * FISH_SIZE_COUNT + sz];
            if (n > 0) {
                RecompuiResource r = row(c2);
                char name[48];
                anyFish = TRUE;
                fate_fish_name(t, sz, name, sizeof(name));
                buf[0] = '\0';
                fate_strcat(buf, name);
                fate_strcat(buf, " x");
                fate_strcat_int(buf, n);
                if (gFishDefs[t].petForm < PET_FORM_COUNT) {
                    fate_strcat(buf, " -> ");
                    fate_strcat(buf, gPetForms[gFishDefs[t].petForm].name);
                } else if (gFishDefs[t].petForm == PET_FORM_REVERT) {
                    fate_strcat(buf, " -> true form");
                }
                text(r, buf, 17.0f, sz == FISH_SIZE_FLAWLESS ? MSG_COLOR_GOLD : UI_TEXT);
                if (gFishDefs[t].petForm != PET_FORM_NONE) {
                    button(r, "Feed", ACT_FEED_FISH, (t << 4) | sz, FALSE);
                }
            }
        }
    }
    if (!anyFish) {
        text(c2, "No fish. Buy a pole from Captain Blubber in Grove, wade into\nwater and press L. Small fish last 2 minutes, Flawless forever.", 16.0f, UI_TEXT_DIM);
    }

    header(c3, "Pet's bag");
    for (i = 0; i < PET_BAG_SIZE; i++) {
        if (!fate_item_empty(&gFate.petBag[i])) {
            item_button(c3, &gFate.petBag[i], SEL_PETBAG, i, NULL);
        }
    }
    if (bagCount == 0) {
        text(c3, "Empty. When your pack is full, loot goes to your pet.", 16.0f, UI_TEXT_DIM);
    }
    if ((sSelKind == SEL_PETBAG || sSelKind == SEL_PETEQUIP) && selected_item() && !fate_item_empty(selected_item())) {
        item_details(c3, selected_item());
        button(c3, sSelKind == SEL_PETBAG ? "Take back" : "Remove from pet", sSelKind == SEL_PETBAG ? ACT_PET_TAKE : ACT_PET_UNEQUIP, sSelIndex, TRUE);
    }
}

/* ------------------------------------------------------------------ */
/* Spells tab                                                          */
/* ------------------------------------------------------------------ */

static void build_spells(RecompuiResource body) {
    RecompuiResource c = scroller(body, 0.0f, 720.0f);
    char buf[160];
    s32 i;

    header(c, "Spellbook (D-Pad Right casts, L cycles)");
    for (i = 0; i < SPELL_COUNT; i++) {
        const FateSpellDef *d = &gSpellDefs[i];
        RecompuiResource r = row(c);
        RecompuiResource info = col(r, 0.0f);
        buf[0] = '\0';
        fate_strcat(buf, d->name);
        fate_strcat(buf, "  (");
        fate_strcat(buf, gSkillNames[d->school]);
        fate_strcat(buf, ", needs ");
        fate_strcat_int(buf, d->reqMag);
        fate_strcat(buf, " Magic)");
        text(info, buf, 19.0f, fate_spell_known(i) ? (i == gFate.selectedSpell ? UI_HEADER : UI_TEXT) : UI_TEXT_DIM);
        buf[0] = '\0';
        fate_strcat(buf, d->desc);
        if (fate_spell_known(i)) {
            fate_strcat(buf, "  Costs ");
            fate_strcat_int(buf, fate_spell_mana_cost(i));
            fate_strcat(buf, " mana.");
        } else {
            fate_strcat(buf, "  Sold by Cheato in Grove.");
        }
        text(info, buf, 16.0f, UI_TEXT_DIM);
        if (fate_spell_known(i)) {
            button(r, i == gFate.selectedSpell ? "Selected" : "Select", ACT_SELECT_SPELL, i, i == gFate.selectedSpell);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Quests tab                                                          */
/* ------------------------------------------------------------------ */

static void quest_card(RecompuiResource parent, const FateQuest *q, s32 index, bool board) {
    RecompuiResource card = inset(parent, 0.0f);
    char buf[256];
    RecompuiResource r;
    fate_quest_title(q, buf, sizeof(buf));
    text(card, buf, 19.0f, q->state == QSTATE_READY ? MSG_COLOR_XP : MSG_COLOR_QUEST);
    fate_quest_describe(q, buf, sizeof(buf));
    text(card, buf, 16.0f, UI_TEXT);
    r = row(card);
    if (board) {
        button(r, "Accept", ACT_QUEST_ACCEPT, index, TRUE);
    } else {
        if (q->state == QSTATE_READY && fate_in_town()) {
            button(r, "Collect reward", ACT_QUEST_TURNIN, index, TRUE);
        }
        button(r, "Abandon", ACT_QUEST_ABANDON, index, FALSE);
    }
}

static void build_quests(RecompuiResource body) {
    RecompuiResource c1 = col(body, 0.0f);
    RecompuiResource c2 = col(body, 0.0f);
    s32 i;
    bool any = FALSE;

    header(c1, "Quest log");
    for (i = 0; i < MAX_ACTIVE_QUESTS; i++) {
        if (gFate.quests[i].type != QUEST_NONE) {
            quest_card(c1, &gFate.quests[i], i, FALSE);
            any = TRUE;
        }
    }
    if (!any) {
        text(c1, "No quests. The folk of Grove post requests on the quest board.", 17.0f, UI_TEXT_DIM);
    }

    header(c2, "Grove quest board");
    if (!fate_in_town()) {
        text(c2, "Return to Grove (Spiral Mountain) to take on new quests\nand collect your rewards.", 17.0f, UI_TEXT_DIM);
        return;
    }
    for (i = 0; i < QUEST_BOARD_SIZE; i++) {
        if (gFate.board[i].type != QUEST_NONE) {
            quest_card(c2, &gFate.board[i], i, TRUE);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Grove tab                                                           */
/* ------------------------------------------------------------------ */

static void price_row(RecompuiResource parent, const char *label, s32 price, s32 action, s32 arg) {
    RecompuiResource r = row(parent);
    char buf[96];
    text(r, label, 18.0f, UI_TEXT);
    buf[0] = '\0';
    fate_strcat(buf, "Buy (");
    fate_strcat_int(buf, price);
    fate_strcat(buf, "g)");
    button(r, buf, action, arg, gFate.gold >= (u32)price);
}

static void build_grove(RecompuiResource body) {
    RecompuiResource nav = inset(body, 320.0f);
    RecompuiResource c = scroller(body, 0.0f, 720.0f);
    char buf[160];
    s32 i;

    header(nav, "Grove");
    for (i = 0; i < GROVE_COUNT; i++) {
        button(nav, sGroveNames[i], ACT_GROVE_SECTION, i, i == sGrove);
    }
    if (gFate.portalMap != 0) {
        button(nav, "Step through the portal", ACT_PORTAL, 0, FALSE);
    }
    buf[0] = '\0';
    fate_strcat(buf, "Gold: ");
    fate_strcat_int(buf, (s32)gFate.gold);
    text(nav, buf, 20.0f, MSG_COLOR_GOLD);

    header(c, sGroveNames[sGrove]);
    switch (sGrove) {
        case GROVE_BOTTLES:
            text(c, "Bottles: \"Potions, scrolls, and I'll identify your finds, Banjo!\"", 17.0f, UI_TEXT_DIM);
            for (i = 0; i < CONS_COUNT; i++) {
                buf[0] = '\0';
                fate_strcat(buf, gConsumableNames[i]);
                fate_strcat(buf, "  (have ");
                fate_strcat_int(buf, gFate.consumables[i]);
                fate_strcat(buf, ")");
                price_row(c, buf, gConsumablePrices[i], ACT_BUY_CONS, i);
            }
            header(c, "Identify");
            {
                bool any = FALSE;
                for (i = 0; i < INV_SIZE; i++) {
                    if (!fate_item_empty(&gFate.inv[i]) && !fate_item_identified(&gFate.inv[i])) {
                        char name[96];
                        RecompuiResource r = row(c);
                        any = TRUE;
                        fate_item_name(&gFate.inv[i], name, sizeof(name));
                        text(r, name, 17.0f, fate_rarity_color(gFate.inv[i].rarity));
                        buf[0] = '\0';
                        fate_strcat(buf, "Identify (");
                        fate_strcat_int(buf, fate_identify_cost(&gFate.inv[i]));
                        fate_strcat(buf, "g)");
                        button(r, buf, ACT_IDENTIFY_PAID, i, FALSE);
                    }
                }
                if (!any) {
                    text(c, "Nothing to identify.", 16.0f, UI_TEXT_DIM);
                }
            }
            header(c, "Sell");
            text(c, "Sell single items from the Inventory tab while you're in Grove.", 16.0f, UI_TEXT_DIM);
            button(c, "Sell all plain (white) items", ACT_SELL_JUNK, 0, FALSE);
            break;

        case GROVE_GOBI: {
            FateItem *sel = (sSelKind == SEL_SHOP) ? selected_item() : NULL;
            text(c, "Gobi: \"Lugged it all the way from the desert. New stock every visit.\"", 17.0f, UI_TEXT_DIM);
            for (i = 0; i < SHOP_STOCK; i++) {
                if (!fate_item_empty(&gShopStock[i])) {
                    RecompuiResource r = row(c);
                    char name[96];
                    fate_item_name(&gShopStock[i], name, sizeof(name));
                    buf[0] = '\0';
                    fate_strcat(buf, name);
                    fate_strcat(buf, "  - ");
                    fate_strcat_int(buf, fate_item_buy_price(&gShopStock[i]));
                    fate_strcat(buf, "g");
                    {
                        RecompuiResource b = button(r, buf, ACT_SELECT, (SEL_SHOP << 8) | i, sSelKind == SEL_SHOP && sSelIndex == i);
                        ui_set_text_color(b, fate_rarity_color(gShopStock[i].rarity));
                    }
                    button(r, "Buy", ACT_BUY_ITEM, i, FALSE);
                }
            }
            if (sel != NULL && !fate_item_empty(sel)) {
                item_details(c, sel);
            }
            break;
        }

        case GROVE_CHEATO:
            text(c, "Cheato: \"Squawk! Knowledge isn't free, bear. Neither am I.\"", 17.0f, UI_TEXT_DIM);
            for (i = 0; i < SPELL_COUNT; i++) {
                if (!fate_spell_known(i)) {
                    buf[0] = '\0';
                    fate_strcat(buf, gSpellDefs[i].name);
                    fate_strcat(buf, " (needs ");
                    fate_strcat_int(buf, gSpellDefs[i].reqMag);
                    fate_strcat(buf, " Magic) - ");
                    fate_strcat(buf, gSpellDefs[i].desc);
                    price_row(c, buf, gSpellDefs[i].price, ACT_BUY_SPELL, i);
                }
            }
            break;

        case GROVE_MUMBO: {
            FateItem *sel = selected_item();
            text(c, "Mumbo: \"Mumbo add magic to item. Sometimes... it go wrong.\"\nEnchanting: 60% new or stronger magic, 25% nothing, 10% all magic lost, 5% curse.", 16.0f, UI_TEXT_DIM);
            for (i = 0; i < SLOT_COUNT; i++) {
                if (!fate_item_empty(&gFate.equip[i])) {
                    item_button(c, &gFate.equip[i], SEL_EQUIP, i, "(worn) ");
                }
            }
            for (i = 0; i < INV_SIZE; i++) {
                if (!fate_item_empty(&gFate.inv[i])) {
                    item_button(c, &gFate.inv[i], SEL_INV, i, NULL);
                }
            }
            if ((sSelKind == SEL_INV || sSelKind == SEL_EQUIP) && sel != NULL && !fate_item_empty(sel)) {
                RecompuiResource r;
                item_details(c, sel);
                r = row(c);
                buf[0] = '\0';
                fate_strcat(buf, "Enchant (");
                fate_strcat_int(buf, fate_enchant_cost(sel));
                fate_strcat(buf, "g)");
                button(r, buf, ACT_ENCHANT, 0, TRUE);
                buf[0] = '\0';
                fate_strcat(buf, "Add socket (");
                fate_strcat_int(buf, fate_socket_cost(sel));
                fate_strcat(buf, "g)");
                button(r, buf, ACT_SOCKET_ADD, 0, FALSE);
            }
            break;
        }

        case GROVE_BRENTILDA:
            text(c, "Brentilda: \"My nasty sister's minions treating you rough, dearie?\nI'll patch you and your little friend up, free of charge.\"", 17.0f, UI_TEXT_DIM);
            button(c, "Heal Banjo, Kazooie and the pet", ACT_HEAL, 0, TRUE);
            break;

        case GROVE_TOOTY:
            text(c, "Tooty: \"I'll tell everyone how brave my big brother is!\nFame unlocks Elite and Legendary gear, and every fame title grants 4 skill points.\"", 17.0f, UI_TEXT_DIM);
            buf[0] = '\0';
            fate_strcat(buf, "Spread tales of your deeds (+25 fame)");
            price_row(c, buf, fate_fame_cost(), ACT_BUY_FAME, 0);
            break;

        case GROVE_BLUBBER:
            text(c, "Captain Blubber: \"Har! Fish the waters where ye wade, and feed yer\nlittle friend the catch. A flawless fish changes it for good!\"", 17.0f, UI_TEXT_DIM);
            if (gFate.poleTier < 3) {
                static const char *poles[3] = { "Fishing Pole", "Sturdy Fishing Pole (bigger fish)", "Blubber's Golden Rod (best fish)" };
                price_row(c, poles[gFate.poleTier], fate_pole_price(), ACT_BUY_POLE, 0);
            } else {
                text(c, "You own the Golden Rod.", 17.0f, MSG_COLOR_GOLD);
            }
            header(c, "Live fish (medium)");
            for (i = 0; i <= FISH_DOGFISH; i++) {
                if (i == FISH_BIGBUTT_BASS && fate_depth_for_level(LEVEL_3_CLANKERS_CAVERN) > 0 && gFate.questsCompleted < 3) {
                    continue; /* rare stock unlocks once the town trusts you */
                }
                buf[0] = '\0';
                fate_strcat(buf, gFishDefs[i].name);
                price_row(c, buf, fate_fish_price(i, FISH_SIZE_MEDIUM) * 3, ACT_BUY_FISH, i);
            }
            button(c, "Sell all your fish", ACT_SELL_FISH, 0, FALSE);
            break;
    }
}

/* ------------------------------------------------------------------ */
/* Gruntilda's bargain                                                 */
/* ------------------------------------------------------------------ */

static void build_bargain(RecompuiResource body) {
    RecompuiResource c = col(body, 0.0f);
    RecompuiResource h;
    char buf[160];
    recompui_set_align_items(c, ALIGN_ITEMS_CENTER);
    h = text(c, "\"Not so fast, bear!\"", 40.0f, 0x8EE66BFF);
    recompui_set_font_weight(h, 700);
    text(c, "Gruntilda appears in a puff of green smoke as you fall.\n\"Perhaps we can come to an... arrangement. Hee hee!\"", 20.0f, UI_TEXT);
    buf[0] = '\0';
    fate_strcat(buf, "Rise right here. I'll take some of your experience and fame.");
    button(c, buf, ACT_BARGAIN, 0, TRUE);
    buf[0] = '\0';
    fate_strcat(buf, "Rise at this area's entrance. I'll take ");
    fate_strcat_int(buf, (s32)(gFate.gold / 10));
    fate_strcat(buf, " of your gold.");
    button(c, buf, ACT_BARGAIN, 1, FALSE);
    buf[0] = '\0';
    fate_strcat(buf, "Flee to Grove and leave your ");
    fate_strcat_int(buf, (s32)gFate.gold);
    fate_strcat(buf, " gold behind (back within 5 minutes to reclaim it).");
    button(c, buf, ACT_BARGAIN, 2, FALSE);
    button(c, "Refuse her. Accept your fate (a normal Banjo-Kazooie death).", ACT_BARGAIN, 3, FALSE);
}

/* ------------------------------------------------------------------ */
/* Build / show                                                        */
/* ------------------------------------------------------------------ */

static void rebuild(void) {
    RecompuiResource tabs, r;
    char buf[128];
    s32 i;

    if (sBody != RECOMPUI_NULL_RESOURCE) {
        recompui_destroy_element(sPanel, sBody);
    }
    sBody = ui_box(sCtx, sPanel, FALSE);
    recompui_set_gap(sBody, 10.0f, UNIT_DP);
    recompui_set_flex_grow(sBody, 1.0f);

    if (gFateBargainOpen) {
        build_bargain(sBody);
        return;
    }

    r = row(sBody);
    recompui_set_justify_content(r, JUSTIFY_CONTENT_SPACE_BETWEEN);
    buf[0] = '\0';
    fate_strcat(buf, "Banjo and Kazooie  -  Level ");
    fate_strcat_int(buf, gFate.level);
    fate_strcat(buf, "  -  ");
    fate_strcat(buf, fate_fame_title());
    {
        RecompuiResource t = text(r, buf, 26.0f, UI_HEADER);
        recompui_set_font_weight(t, 700);
    }
    button(r, "Close", ACT_CLOSE, 0, FALSE);

    tabs = row(sBody);
    for (i = 0; i < TAB_COUNT; i++) {
        if (i == TAB_GROVE && !fate_in_town()) {
            continue;
        }
        button(tabs, sTabNames[i], ACT_TAB, i, i == sTab);
    }

    if (sNotice[0]) {
        text(sBody, sNotice, 18.0f, MSG_COLOR_QUEST);
    }

    r = row(sBody);
    recompui_set_align_items(r, ALIGN_ITEMS_FLEX_START);
    recompui_set_flex_grow(r, 1.0f);
    switch (sTab) {
        case TAB_CHARACTER: build_character(r); break;
        case TAB_INVENTORY: build_inventory(r); break;
        case TAB_PET:       build_pet(r); break;
        case TAB_SPELLS:    build_spells(r); break;
        case TAB_QUESTS:    build_quests(r); break;
        case TAB_GROVE:     build_grove(r); break;
    }
}

void fate_menu_init(void) {
    sCtx = recompui_create_context();
    recompui_open_context(sCtx);
    sRoot = recompui_context_root(sCtx);
    ui_absolute(sRoot, 0.0f, 0.0f, 0.0f, 0.0f);
    recompui_set_width(sRoot, 100.0f, UNIT_PERCENT);
    recompui_set_height(sRoot, 100.0f, UNIT_PERCENT);
    recompui_set_display(sRoot, DISPLAY_FLEX);
    recompui_set_justify_content(sRoot, JUSTIFY_CONTENT_CENTER);
    recompui_set_align_items(sRoot, ALIGN_ITEMS_CENTER);
    ui_set_bg(sRoot, 0x00000099);

    sPanel = ui_box(sCtx, sRoot, FALSE);
    recompui_set_width(sPanel, 1500.0f, UNIT_DP);
    recompui_set_height(sPanel, 940.0f, UNIT_DP);
    recompui_set_max_width(sPanel, 96.0f, UNIT_PERCENT);
    ui_set_bg(sPanel, UI_PANEL_BG);
    ui_set_border(sPanel, UI_PANEL_EDGE, 3.0f, 16.0f);
    recompui_set_padding(sPanel, 20.0f, UNIT_DP);
    sBody = RECOMPUI_NULL_RESOURCE;
    recompui_close_context(sCtx);
}

static void show(void) {
    if (!sShown) {
        sShown = TRUE;
        gFateMenuOpen = TRUE;
        sWorldWasEnabled = func_8033511C();
        func_80335110(0); /* freeze the world, exactly like the pause menu */
        recompui_show_context(sCtx);
        fate_play_sfx(SFX_C5_TWINKLY_POP);
    }
    sRebuild = TRUE;
}

void fate_menu_open(s32 tab) {
    if (sCtx == RECOMPUI_NULL_CONTEXT || !fate_gameplay_active()) {
        return;
    }
    sTab = tab;
    if (sTab == TAB_GROVE && !fate_in_town()) {
        sTab = TAB_CHARACTER;
    }
    sSelKind = SEL_NONE;
    sRetireMode = FALSE;
    sNotice[0] = '\0';
    fate_recalc();
    show();
}

void fate_bargain_open(void) {
    if (sCtx == RECOMPUI_NULL_CONTEXT) {
        return;
    }
    gFateBargainOpen = TRUE;
    fate_play_sfx(SFX_EA_GRUNTY_LAUGH_1);
    show();
}

void fate_menu_close(void) {
    if (!sShown) {
        return;
    }
    sShown = FALSE;
    gFateMenuOpen = FALSE;
    recompui_hide_context(sCtx);
    func_80335110(sWorldWasEnabled ? 1 : 0);
    fate_hud_mark_dirty();
}

void fate_menu_request_rebuild(void) {
    sRebuild = TRUE;
}

void fate_menu_frame(void) {
    if (sCtx == RECOMPUI_NULL_CONTEXT || !sShown) {
        return;
    }
    if (sRebuild) {
        sRebuild = FALSE;
        recompui_open_context(sCtx);
        rebuild();
        recompui_close_context(sCtx);
    }
}

/* ------------------------------------------------------------------ */
/* Actions                                                             */
/* ------------------------------------------------------------------ */

static void do_action(s32 action, s32 arg) {
    FateItem *sel = selected_item();
    char reason[64];

    switch (action) {
        case ACT_TAB:
            sTab = arg;
            sSelKind = SEL_NONE;
            sNotice[0] = '\0';
            break;
        case ACT_CLOSE:
            fate_menu_close();
            return;
        case ACT_RAISE_STAT:
            fate_raise_stat(arg);
            break;
        case ACT_RAISE_SKILL:
            fate_raise_skill(arg);
            break;
        case ACT_SELECT:
            sSelKind = (arg >> 8) & 0xFF;
            sSelIndex = arg & 0xFF;
            break;
        case ACT_EQUIP:
            if (fate_item_can_equip(&gFate.inv[arg], reason, sizeof(reason))) {
                fate_equip_from_inventory(arg);
                sSelKind = SEL_NONE;
                notice("Equipped.");
            } else {
                notice(reason);
            }
            break;
        case ACT_UNEQUIP:
            if (fate_unequip(arg)) {
                sSelKind = SEL_NONE;
            } else {
                notice("Your pack is full.");
            }
            break;
        case ACT_GIVE_PET:
            if (fate_pet_bag_add(&gFate.inv[arg])) {
                fate_item_clear(&gFate.inv[arg]);
                sSelKind = SEL_NONE;
            } else {
                notice("Your pet's bag is full.");
            }
            break;
        case ACT_DROP:
            if (gFate.inv[arg].rarity == RARITY_ARTIFACT) {
                notice("You can't drop a quest artifact.");
            } else {
                fate_item_clear(&gFate.inv[arg]);
                sSelKind = SEL_NONE;
            }
            break;
        case ACT_SELL:
            fate_shop_sell_inventory(arg);
            sSelKind = SEL_NONE;
            break;
        case ACT_IDENTIFY_SCROLL:
            if (gFate.consumables[CONS_IDENTIFY] > 0) {
                gFate.consumables[CONS_IDENTIFY]--;
                fate_identify(&gFate.inv[arg]);
                fate_play_sfx(SFX_A6_MAGICAL_FINISH);
            }
            break;
        case ACT_SOCKET_GEM:
            if (sel != NULL && fate_socket_gem(sel, arg)) {
                notice("The gem settles into its socket.");
            }
            break;
        case ACT_USE_RETURN:
            if (gFate.consumables[CONS_TOWN_PORTAL] > 0) {
                fate_menu_close();
                if (fate_warp_to_town()) {
                    gFate.consumables[CONS_TOWN_PORTAL]--;
                }
                return;
            }
            break;
        case ACT_PET_EQUIP: {
            s32 slot = fate_item_base(&gFate.inv[arg])->kind == KIND_AMULET ? 0 : 1;
            FateItem tmp = gFate.petEquip[slot];
            if (!fate_item_identified(&gFate.inv[arg])) {
                notice("Identify it first.");
                break;
            }
            gFate.petEquip[slot] = gFate.inv[arg];
            gFate.inv[arg] = tmp;
            sSelKind = SEL_NONE;
            notice("Your pet wears it proudly.");
            break;
        }
        case ACT_PET_UNEQUIP:
            if (fate_inventory_add(&gFate.petEquip[arg])) {
                fate_item_clear(&gFate.petEquip[arg]);
                sSelKind = SEL_NONE;
            } else {
                notice("Your pack is full.");
            }
            break;
        case ACT_PET_TAKE:
            if (fate_inventory_add(&gFate.petBag[arg])) {
                fate_item_clear(&gFate.petBag[arg]);
                sSelKind = SEL_NONE;
            } else {
                notice("Your pack is full.");
            }
            break;
        case ACT_PET_SEND:
            fate_pet_send_to_town();
            break;
        case ACT_PET_POTION:
            if (gFate.consumables[CONS_HP_POTION] > 0) {
                gFate.consumables[CONS_HP_POTION]--;
                gFate.petHp = (u16)fate_mini(fate_pet_max_hp(), gFate.petHp + fate_pet_max_hp() / 2);
                notice("Your pet laps up the potion.");
            }
            break;
        case ACT_FEED_FISH:
            fate_pet_feed_fish(arg >> 4, arg & 0xF);
            break;
        case ACT_SELECT_SPELL:
            gFate.selectedSpell = (u8)arg;
            break;
        case ACT_QUEST_ACCEPT:
            fate_quest_accept(arg);
            break;
        case ACT_QUEST_TURNIN:
            fate_quest_turn_in(arg, FALSE);
            break;
        case ACT_QUEST_ABANDON:
            fate_quest_abandon(arg);
            break;
        case ACT_GROVE_SECTION:
            sGrove = arg;
            sSelKind = SEL_NONE;
            sNotice[0] = '\0';
            break;
        case ACT_BUY_CONS:
            fate_shop_buy_consumable(arg);
            break;
        case ACT_BUY_ITEM:
            fate_shop_buy_item(arg);
            sSelKind = SEL_NONE;
            break;
        case ACT_BUY_SPELL:
            fate_shop_buy_spell(arg);
            break;
        case ACT_IDENTIFY_PAID:
            fate_shop_identify(arg);
            break;
        case ACT_SELL_JUNK:
            if (fate_shop_sell_junk() == 0) {
                notice("Bottles: \"Nothing plain to sell, Banjo.\"");
            }
            break;
        case ACT_ENCHANT:
            if (sel != NULL) {
                notice(fate_enchant(sel));
            }
            break;
        case ACT_SOCKET_ADD:
            if (sel != NULL) {
                notice(fate_add_socket(sel));
            }
            break;
        case ACT_HEAL:
            fate_healer();
            break;
        case ACT_BUY_FAME:
            fate_buy_fame();
            break;
        case ACT_BUY_POLE:
            fate_shop_buy_pole();
            break;
        case ACT_BUY_FISH:
            fate_shop_buy_fish(arg);
            break;
        case ACT_SELL_FISH:
            if (fate_shop_sell_all_fish() == 0) {
                notice("Captain Blubber: \"Ye've got no fish, matey!\"");
            }
            break;
        case ACT_PORTAL:
            fate_menu_close();
            fate_warp_to_portal();
            return;
        case ACT_RETIRE_MODE:
            sRetireMode = !sRetireMode;
            break;
        case ACT_RETIRE_ITEM:
            fate_retire(arg);
            sRetireMode = FALSE;
            break;
        case ACT_NIGHTMARE:
            if (gFate.nightmare < 5) {
                gFate.nightmare++;
                notice("The worlds grow darker. Every monster is now ten depths stronger.");
            }
            break;
        case ACT_BARGAIN:
            gFateBargainOpen = FALSE;
            fate_menu_close();
            fate_bargain_resolve(arg);
            return;
    }
    fate_hud_mark_dirty();
    sRebuild = TRUE;
}

static void on_click(RecompuiResource resource, const RecompuiEventData *event, void *userdata) {
    u32 data = (u32)userdata;
    if (event->type != UI_EVENT_CLICK) {
        return;
    }
    do_action((s32)(data >> 16), (s32)(data & 0xFFFF));
}
