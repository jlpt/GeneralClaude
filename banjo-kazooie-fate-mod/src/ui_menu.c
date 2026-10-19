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
 * RmlUi updates and lays out every document on the render thread without
 * taking the context lock, so creating or destroying elements while any
 * context is on screen can crash the game. The menu therefore creates a
 * fixed pool of elements once, while nothing is on screen, and afterwards
 * only changes text (recompui queues that for the render thread) and
 * overwrites properties that were all set when the pool was created.
 *
 * Each page fills up to three columns of lines. A line is a label and up to
 * two buttons. The build functions describe the page with put_text,
 * put_button and friends, then apply_page pushes only what changed to the
 * pool. Button callbacks look their action up in the page description.
 */

#define MENU_COLS     3
#define COL_LINES     44
#define LINE_BUTTONS  2
#define LINE_TEXT     160
#define BUTTON_TEXT   80

/* Click ids for the buttons outside the columns. Column buttons use their pool index. */
#define CLICK_CLOSE   0xF000
#define CLICK_TAB     0xF100

#define BTN_BG        0x3A2C1EFF
#define BTN_EDGE      0x8A6A35FF
#define BTN_BG_ON     0x7A5418FF
#define BTN_EDGE_ON   0xF2C744FF
#define GRUNTY_GREEN  0x8EE66BFF

enum menu_action {
    ACT_NONE,
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

/* Short slot labels, so item names fit on one line in the equipment list. */
static const char *sSlotShort[SLOT_COUNT] = { "Paws", "Beak", "Helm", "Armor", "Pack", "Boots", "Amulet", "Ring", "Ring" };

enum line_style { LS_NORMAL, LS_SMALL, LS_HEADER, LS_COUNT };

static const f32 sStyleSize[LS_COUNT] = { 18.0f, 16.0f, 23.0f };
static const u32 sStyleWeight[LS_COUNT] = { 400, 400, 700 };

/* ------------------------------------------------------------------ */
/* Page description                                                    */
/* ------------------------------------------------------------------ */

typedef struct {
    char text[BUTTON_TEXT];
    u32 color;
    s16 action;
    s16 arg;
    u8 primary;
    u8 wide;
} ButtonSpec;

typedef struct {
    char text[LINE_TEXT];
    u32 color;
    u8 style;
    u8 gap;     /* extra space above the line */
    u8 closed;  /* no more buttons may join this line */
    u8 buttons;
    ButtonSpec btn[LINE_BUTTONS];
} LineSpec;

static LineSpec sSpec[MENU_COLS][COL_LINES];
static s32 sSpecCount[MENU_COLS];
static f32 sSpecWidth[MENU_COLS];   /* <0 hidden, 0 fills the remaining width, >0 fixed width */
static bool sSpecOverflow[MENU_COLS];
static s32 sCol;
static bool sPendingGap;
static char sTitle[96];
static u32 sTitleColor;

/* ------------------------------------------------------------------ */
/* Element pool                                                        */
/* ------------------------------------------------------------------ */

typedef struct {
    RecompuiResource el;
    RecompuiResource label;
    u32 hash;
    u32 color;
    u8 shown;
    u8 primary;
    u8 wide;
} PoolButton;

typedef struct {
    RecompuiResource row;
    RecompuiResource label;
    u32 hash;
    u32 color;
    u8 style;
    u8 gap;
    u8 shown;
    u8 labelShown;
    PoolButton btn[LINE_BUTTONS];
} PoolLine;

typedef struct {
    RecompuiResource box;
    f32 width;
    PoolLine lines[COL_LINES];
} PoolColumn;

static RecompuiContext sCtx = RECOMPUI_NULL_CONTEXT;
static bool sPoolBuilt;
static PoolColumn sCols[MENU_COLS];
static RecompuiResource sTitleLabel;
static RecompuiResource sTabRow;
static RecompuiResource sNoticeLabel;
static PoolButton sCloseBtn;
static PoolButton sTabBtn[TAB_COUNT];
static u32 sTitleHash, sTitleShownColor, sNoticeHash;
static u8 sTabRowShown, sNoticeShown;

/* Menu state */
static bool sShown;
static bool sRebuild;
static bool sRetireMode;
static s32 sTab;
static s32 sGrove;
static s32 sSelKind;
static s32 sSelIndex;
static s32 sWorldWasEnabled;
static char sNotice[128];

static void on_click(RecompuiResource resource, const RecompuiEventData *event, void *userdata);

/* ------------------------------------------------------------------ */
/* Text helpers                                                        */
/* ------------------------------------------------------------------ */

static void copy_text(char *dst, const char *src, s32 len, s32 cap) {
    s32 i;
    if (len > cap - 1) {
        len = cap - 1;
    }
    for (i = 0; i < len && src[i]; i++) {
        dst[i] = src[i];
    }
    dst[i] = '\0';
}

/* Never 0, so every pool element gets its first text. */
static u32 text_hash(const char *s) {
    u32 h = 2166136261u;
    while (*s) {
        h = (h ^ (u8)*s++) * 16777619u;
    }
    return h | 1;
}

static void notice(const char *msg) {
    copy_text(sNotice, msg, 127, sizeof(sNotice));
}

/* ------------------------------------------------------------------ */
/* Describing a page                                                   */
/* ------------------------------------------------------------------ */

static void page_begin(void) {
    s32 c;
    for (c = 0; c < MENU_COLS; c++) {
        sSpecCount[c] = 0;
        sSpecWidth[c] = -1.0f;
        sSpecOverflow[c] = FALSE;
    }
    sCol = 0;
    sPendingGap = FALSE;
}

static void use_col(s32 c, f32 width) {
    sCol = c;
    sSpecWidth[c] = width;
    sPendingGap = FALSE;
}

static LineSpec *new_line(void) {
    s32 n = sSpecCount[sCol];
    LineSpec *l;
    if (n >= COL_LINES) {
        sSpecOverflow[sCol] = TRUE;
        return NULL;
    }
    l = &sSpec[sCol][n];
    sSpecCount[sCol] = n + 1;
    l->text[0] = '\0';
    l->color = UI_TEXT;
    l->style = LS_NORMAL;
    l->gap = sPendingGap && n > 0;
    l->closed = FALSE;
    l->buttons = 0;
    sPendingGap = FALSE;
    return l;
}

static LineSpec *last_line(void) {
    s32 n = sSpecCount[sCol];
    return n > 0 ? &sSpec[sCol][n - 1] : NULL;
}

static void end_line(void) {
    LineSpec *l = last_line();
    if (l != NULL) {
        l->closed = TRUE;
    }
}

static void put_gap(void) {
    end_line();
    sPendingGap = TRUE;
}

/* One line per '\n'-separated part. The last line stays open for buttons. */
static void put_text(const char *text, u32 color, s32 style) {
    while (TRUE) {
        s32 n = 0;
        LineSpec *l;
        while (text[n] != '\0' && text[n] != '\n') {
            n++;
        }
        l = new_line();
        if (l != NULL) {
            copy_text(l->text, text, n, LINE_TEXT);
            l->color = color;
            l->style = style;
        }
        if (text[n] == '\0') {
            break;
        }
        end_line();
        text += n + 1;
    }
}

static void header(const char *text) {
    put_gap();
    put_text(text, UI_HEADER, LS_HEADER);
    end_line();
}

static void put_line(const char *text, u32 color, s32 style) {
    put_text(text, color, style);
    end_line();
}

static ButtonSpec *add_button(const char *text, s32 action, s32 arg, bool primary, bool ownLine) {
    LineSpec *l = last_line();
    ButtonSpec *b;
    if (ownLine || l == NULL || l->closed || l->buttons >= LINE_BUTTONS) {
        l = new_line();
        if (l == NULL) {
            return NULL;
        }
    }
    b = &l->btn[l->buttons++];
    copy_text(b->text, text, BUTTON_TEXT, BUTTON_TEXT);
    b->color = UI_TEXT;
    b->action = action;
    b->arg = arg;
    b->primary = primary;
    b->wide = FALSE;
    return b;
}

/* Joins the current line (next to its text or a previous button) when there is room. */
static void put_button(const char *text, s32 action, s32 arg, bool primary) {
    add_button(text, action, arg, primary, FALSE);
}

/* A full-width, left-aligned button on its own line. One more put_button may join it. */
static void put_wide(const char *text, u32 color, s32 action, s32 arg, bool primary) {
    ButtonSpec *b = add_button(text, action, arg, primary, TRUE);
    if (b != NULL) {
        b->color = color;
        b->wide = TRUE;
    }
}

static void put_stat(const char *label, s32 value, const char *suffix) {
    char buf[96];
    buf[0] = '\0';
    fate_strcat(buf, label);
    fate_strcat(buf, ": ");
    fate_strcat_int(buf, value);
    fate_strcat(buf, suffix);
    put_line(buf, UI_TEXT, LS_NORMAL);
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

static void put_item(const FateItem *item, s32 selKind, s32 index, const char *prefix) {
    char name[96];
    char line[BUTTON_TEXT];
    fate_item_name(item, name, sizeof(name));
    line[0] = '\0';
    if (prefix) {
        fate_strcat(line, prefix);
    }
    copy_text(line + fate_strlen(line), name, BUTTON_TEXT, BUTTON_TEXT - fate_strlen(line));
    put_wide(line, fate_rarity_color(item->rarity), ACT_SELECT, (selKind << 8) | index, sSelKind == selKind && sSelIndex == index);
}

static void put_item_details(const FateItem *item) {
    char name[96];
    static char desc[768];
    if (item == NULL || fate_item_empty(item)) {
        put_line("Select an item to see its details.", UI_TEXT_DIM, LS_SMALL);
        return;
    }
    fate_item_name(item, name, sizeof(name));
    put_line(name, fate_rarity_color(item->rarity), LS_HEADER);
    fate_item_describe(item, desc, sizeof(desc));
    put_line(desc, UI_TEXT, LS_SMALL);
}

static void put_price(const char *label, s32 price, s32 action, s32 arg) {
    char buf[32];
    buf[0] = '\0';
    fate_strcat(buf, "Buy (");
    fate_strcat_int(buf, price);
    fate_strcat(buf, "g)");
    put_text(label, UI_TEXT, LS_NORMAL);
    put_button(buf, action, arg, gFate.gold >= (u32)price);
    end_line();
}

/* ------------------------------------------------------------------ */
/* Character tab                                                       */
/* ------------------------------------------------------------------ */

static void build_character(void) {
    char buf[128];
    s32 i;
    u32 nextFame;

    use_col(0, 440.0f);
    header("Attributes");
    buf[0] = '\0';
    fate_strcat(buf, "Unspent points: ");
    fate_strcat_int(buf, gFate.statPoints);
    put_line(buf, gFate.statPoints ? MSG_COLOR_XP : UI_TEXT_DIM, LS_NORMAL);
    for (i = 0; i < STAT_COUNT; i++) {
        buf[0] = '\0';
        fate_strcat(buf, gStatNames[i]);
        fate_strcat(buf, "  ");
        fate_strcat_int(buf, gDerived.stat[i]);
        if (gDerived.stat[i] != gFate.stats[i]) {
            fate_strcat(buf, " (base ");
            fate_strcat_int(buf, gFate.stats[i]);
            fate_strcat(buf, ")");
        }
        put_text(buf, UI_TEXT, LS_NORMAL);
        if (gFate.statPoints > 0) {
            put_button("+", ACT_RAISE_STAT, i, TRUE);
        }
        end_line();
    }
    header("Combat");
    put_stat("Health", gDerived.maxHp, "");
    put_stat("Mana", gDerived.maxMana, "");
    put_stat("Armor", gDerived.armor, "");
    buf[0] = '\0';
    fate_strcat(buf, "Paw damage: ");
    fate_strcat_int(buf, gDerived.pawMin + gDerived.stat[STAT_STR] / 4);
    fate_strcat(buf, "-");
    fate_strcat_int(buf, gDerived.pawMax + gDerived.stat[STAT_STR] / 4);
    fate_strcat(buf, "   Beak: ");
    fate_strcat_int(buf, gDerived.beakMin);
    fate_strcat(buf, "-");
    fate_strcat_int(buf, gDerived.beakMax);
    put_line(buf, UI_TEXT, LS_NORMAL);
    put_stat("Critical chance", gDerived.critChance / 100, "%");
    put_stat("Block", gDerived.block, "%");
    put_stat("Dodge", gDerived.dodge, "%");
    if (gDerived.goldFind) put_stat("Gold find", gDerived.goldFind, "%");
    if (gDerived.magicFind) put_stat("Magic find", gDerived.magicFind, "%");
    if (gDerived.lifeSteal) put_stat("Life steal", gDerived.lifeSteal, "%");

    use_col(1, 440.0f);
    header("Skills");
    buf[0] = '\0';
    fate_strcat(buf, "Unspent points: ");
    fate_strcat_int(buf, gFate.skillPoints);
    put_line(buf, gFate.skillPoints ? MSG_COLOR_XP : UI_TEXT_DIM, LS_NORMAL);
    for (i = 0; i < SKILL_COUNT; i++) {
        buf[0] = '\0';
        fate_strcat(buf, gSkillNames[i]);
        fate_strcat(buf, "  ");
        fate_strcat_int(buf, gFate.skills[i]);
        put_text(buf, gFate.skills[i] ? UI_TEXT : UI_TEXT_DIM, LS_NORMAL);
        if (gFate.skillPoints > 0 && gFate.skills[i] < FATE_SKILL_MAX) {
            put_button("+", ACT_RAISE_SKILL, i, TRUE);
        }
        end_line();
    }

    use_col(2, 0.0f);
    header("Banjo and Kazooie");
    buf[0] = '\0';
    fate_strcat(buf, "Level ");
    fate_strcat_int(buf, gFate.level);
    fate_strcat(buf, "   Experience ");
    fate_strcat_int(buf, (s32)gFate.xp);
    fate_strcat(buf, " / ");
    fate_strcat_int(buf, (s32)fate_xp_for_level(gFate.level + 1));
    put_line(buf, UI_TEXT, LS_NORMAL);
    buf[0] = '\0';
    fate_strcat(buf, "Fame: ");
    fate_strcat(buf, fate_fame_title());
    fate_strcat(buf, " (level ");
    fate_strcat_int(buf, fate_fame_level());
    fate_strcat(buf, ")");
    put_line(buf, MSG_COLOR_FAME, LS_NORMAL);
    nextFame = fate_fame_for_level(fate_fame_level() + 1);
    buf[0] = '\0';
    fate_strcat_int(buf, (s32)gFate.fame);
    fate_strcat(buf, " / ");
    fate_strcat_int(buf, (s32)nextFame);
    fate_strcat(buf, " fame to the next title (+4 skill points)");
    put_line(buf, UI_TEXT_DIM, LS_SMALL);
    put_stat("Gold", (s32)gFate.gold, "");
    put_stat("Monsters slain", (s32)gFate.kills, "");
    put_stat("Champions slain", (s32)gFate.championKills, "");
    put_stat("Quests completed", gFate.questsCompleted, "");
    put_stat("Deaths", gFate.deaths, "");
    if (gFate.generation > 0) {
        buf[0] = '\0';
        fate_strcat(buf, "Generation ");
        fate_strcat_int(buf, gFate.generation + 1);
        fate_strcat(buf, " of your line");
        put_line(buf, MSG_COLOR_GOLD, LS_NORMAL);
    }
    if (gFate.nightmare > 0) {
        put_stat("Nightmare tier", gFate.nightmare, " (+10 depth each)");
    }

    if (fileProgressFlag_get(FILEPROG_FC_DEFEAT_GRUNTY)) {
        header("Gruntilda is defeated!");
        if (fate_can_retire()) {
            put_line("Retire this hero and pass one equipped item to your heir.\nThe heirloom grows 25% stronger each generation.", UI_TEXT_DIM, LS_SMALL);
            put_wide(sRetireMode ? "Choose the heirloom below..." : "Retire", UI_TEXT, ACT_RETIRE_MODE, 0, TRUE);
            end_line();
            if (sRetireMode) {
                for (i = 0; i < SLOT_COUNT; i++) {
                    if (!fate_item_empty(&gFate.equip[i])) {
                        char name[96];
                        fate_item_name(&gFate.equip[i], name, sizeof(name));
                        put_wide(name, fate_rarity_color(gFate.equip[i].rarity), ACT_RETIRE_ITEM, i, FALSE);
                        end_line();
                    }
                }
            }
        } else if (gFate.retired) {
            put_line("This hero has retired. Start a new game file to play as the heir.", UI_TEXT_DIM, LS_SMALL);
        }
        if (gFate.nightmare < 5) {
            put_wide("Descend into Nightmare (every world +10 depth)", UI_TEXT, ACT_NIGHTMARE, 0, FALSE);
            end_line();
        }
    }
}

/* ------------------------------------------------------------------ */
/* Inventory tab                                                       */
/* ------------------------------------------------------------------ */

static void build_inventory(void) {
    FateItem *sel;
    char buf[128];
    s32 i;

    use_col(0, 500.0f);
    header("Equipped");
    for (i = 0; i < SLOT_COUNT; i++) {
        char prefix[40];
        prefix[0] = '\0';
        fate_strcat(prefix, sSlotShort[i]);
        fate_strcat(prefix, ": ");
        if (fate_item_empty(&gFate.equip[i])) {
            fate_strcat(prefix, "-");
            put_line(prefix, UI_TEXT_DIM, LS_SMALL);
        } else {
            put_item(&gFate.equip[i], SEL_EQUIP, i, prefix);
            end_line();
        }
    }
    header("Belt");
    for (i = 0; i < CONS_COUNT; i++) {
        buf[0] = '\0';
        fate_strcat(buf, gConsumableNames[i]);
        fate_strcat(buf, " x");
        fate_strcat_int(buf, gFate.consumables[i]);
        put_line(buf, gFate.consumables[i] ? UI_TEXT : UI_TEXT_DIM, LS_SMALL);
    }
    if (gFate.consumables[CONS_TOWN_PORTAL] > 0 && !fate_in_town()) {
        put_wide("Read a Scroll of Return (go to Grove)", UI_TEXT, ACT_USE_RETURN, 0, TRUE);
        end_line();
    }
    {
        bool anyGem = FALSE;
        for (i = 1; i < gGemCount && i < 16; i++) {
            if (gFate.gems[i]) {
                if (!anyGem) {
                    header("Gems");
                    anyGem = TRUE;
                }
                buf[0] = '\0';
                fate_strcat(buf, gGems[i].name);
                fate_strcat(buf, " x");
                fate_strcat_int(buf, gFate.gems[i]);
                put_line(buf, 0x7FFFD4FF, LS_SMALL);
            }
        }
    }

    use_col(1, 450.0f);
    buf[0] = '\0';
    fate_strcat(buf, "Pack (");
    fate_strcat_int(buf, INV_SIZE - fate_inventory_free_slots());
    fate_strcat(buf, "/");
    fate_strcat_int(buf, INV_SIZE);
    fate_strcat(buf, ")");
    header(buf);
    for (i = 0; i < INV_SIZE; i++) {
        if (!fate_item_empty(&gFate.inv[i])) {
            put_item(&gFate.inv[i], SEL_INV, i, NULL);
            end_line();
        }
    }
    if (fate_inventory_free_slots() == INV_SIZE) {
        put_line("Your pack is empty. Slay monsters to find loot!", UI_TEXT_DIM, LS_SMALL);
    }

    use_col(2, 0.0f);
    header("Details");
    sel = selected_item();
    if (sSelKind != SEL_INV && sSelKind != SEL_EQUIP) {
        sel = NULL;
    }
    put_item_details(sel);
    if (sel != NULL && !fate_item_empty(sel)) {
        s32 kind = fate_item_base(sel)->kind;
        put_gap();
        if (sSelKind == SEL_INV) {
            if (kind != KIND_ARTIFACT) {
                put_button("Equip", ACT_EQUIP, sSelIndex, TRUE);
            }
            if (kind == KIND_AMULET || kind == KIND_RING) {
                put_button("Put on pet", ACT_PET_EQUIP, sSelIndex, FALSE);
            }
            put_button("Give to pet", ACT_GIVE_PET, sSelIndex, FALSE);
            if (fate_in_town()) {
                buf[0] = '\0';
                fate_strcat(buf, "Sell (");
                fate_strcat_int(buf, fate_item_sell_price(sel));
                fate_strcat(buf, "g)");
                put_button(buf, ACT_SELL, sSelIndex, FALSE);
            }
            put_button("Drop", ACT_DROP, sSelIndex, FALSE);
            end_line();
            if (!fate_item_identified(sel) && gFate.consumables[CONS_IDENTIFY] > 0) {
                put_button("Read a Scroll of Identify", ACT_IDENTIFY_SCROLL, sSelIndex, TRUE);
                end_line();
            }
        } else {
            put_button("Unequip", ACT_UNEQUIP, sSelIndex, TRUE);
            end_line();
        }
        if (fate_item_sockets(sel) > 0) {
            for (i = 1; i < gGemCount && i < 16; i++) {
                if (gFate.gems[i]) {
                    buf[0] = '\0';
                    fate_strcat(buf, "Socket ");
                    fate_strcat(buf, gGems[i].name);
                    put_button(buf, ACT_SOCKET_GEM, i, FALSE);
                }
            }
            end_line();
        }
    }
}

/* ------------------------------------------------------------------ */
/* Pet tab                                                             */
/* ------------------------------------------------------------------ */

static void build_pet(void) {
    char buf[128];
    s32 i, t, sz;
    bool anyFish = FALSE;
    s32 bagCount = 0;
    FateItem *sel;

    use_col(0, 480.0f);
    if (gFate.petSpecies == PET_SPECIES_NONE) {
        header("No pet");
        put_line("Your pet is disabled in the mod settings.", UI_TEXT_DIM, LS_NORMAL);
        return;
    }
    header(fate_pet_name());
    put_stat("Level", gFate.petLevel, "");
    buf[0] = '\0';
    fate_strcat(buf, "Health: ");
    fate_strcat_int(buf, gFate.petHp);
    fate_strcat(buf, " / ");
    fate_strcat_int(buf, fate_pet_max_hp());
    put_line(buf, UI_TEXT, LS_NORMAL);
    put_stat("Damage per bite", fate_pet_damage(), "");
    put_line(fate_pet_status(), MSG_COLOR_PET, LS_NORMAL);
    if (gFate.petFormSeconds > 0) {
        put_stat("Transformation lasts", gFate.petFormSeconds, " more seconds");
    }
    for (i = 0; i < PET_BAG_SIZE; i++) {
        if (!fate_item_empty(&gFate.petBag[i])) {
            bagCount++;
        }
    }
    if (fate_pet_town_seconds() <= 0.0f) {
        put_gap();
        buf[0] = '\0';
        fate_strcat(buf, "Send to Grove to sell loot (");
        fate_strcat_int(buf, bagCount);
        fate_strcat(buf, " items)");
        put_wide(buf, UI_TEXT, ACT_PET_SEND, 0, TRUE);
        end_line();
        if (gFate.consumables[CONS_HP_POTION] > 0 && gFate.petHp < fate_pet_max_hp()) {
            put_wide("Give the pet a Health Potion", UI_TEXT, ACT_PET_POTION, 0, FALSE);
            end_line();
        }
    }
    header("Pet jewelry");
    for (i = 0; i < PET_EQUIP_COUNT; i++) {
        if (fate_item_empty(&gFate.petEquip[i])) {
            put_line(i == 0 ? "Collar: -" : "Charm: -", UI_TEXT_DIM, LS_SMALL);
        } else {
            put_item(&gFate.petEquip[i], SEL_PETEQUIP, i, i == 0 ? "Collar: " : "Charm: ");
            end_line();
        }
    }
    put_line("Put amulets and rings on your pet from the Inventory tab.", UI_TEXT_DIM, LS_SMALL);

    use_col(1, 480.0f);
    header("Fish (feed to transform)");
    for (t = 0; t < FISH_COUNT; t++) {
        for (sz = 0; sz < FISH_SIZE_COUNT; sz++) {
            s32 n = gFate.fish[t * FISH_SIZE_COUNT + sz];
            if (n > 0) {
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
                put_text(buf, sz == FISH_SIZE_FLAWLESS ? MSG_COLOR_GOLD : UI_TEXT, LS_SMALL);
                if (gFishDefs[t].petForm != PET_FORM_NONE) {
                    put_button("Feed", ACT_FEED_FISH, (t << 4) | sz, FALSE);
                }
                end_line();
            }
        }
    }
    if (!anyFish) {
        put_line("No fish yet. Buy a pole from Captain Blubber in Grove, wade into water and press L.", UI_TEXT_DIM, LS_SMALL);
        put_line("Small fish change your pet for 2 minutes. Flawless fish change it for good.", UI_TEXT_DIM, LS_SMALL);
    }

    use_col(2, 0.0f);
    header("Pet's bag");
    for (i = 0; i < PET_BAG_SIZE; i++) {
        if (!fate_item_empty(&gFate.petBag[i])) {
            put_item(&gFate.petBag[i], SEL_PETBAG, i, NULL);
            end_line();
        }
    }
    if (bagCount == 0) {
        put_line("Empty. When your pack is full, loot goes to your pet.", UI_TEXT_DIM, LS_SMALL);
    }
    sel = selected_item();
    if ((sSelKind == SEL_PETBAG || sSelKind == SEL_PETEQUIP) && sel != NULL && !fate_item_empty(sel)) {
        put_gap();
        put_item_details(sel);
        put_button(sSelKind == SEL_PETBAG ? "Take back" : "Remove from pet",
                   sSelKind == SEL_PETBAG ? ACT_PET_TAKE : ACT_PET_UNEQUIP, sSelIndex, TRUE);
        end_line();
    }
}

/* ------------------------------------------------------------------ */
/* Spells tab                                                          */
/* ------------------------------------------------------------------ */

static void build_spells(void) {
    char buf[160];
    s32 i;

    use_col(0, 0.0f);
    header("Spellbook (D-Pad Right casts, L cycles)");
    for (i = 0; i < SPELL_COUNT; i++) {
        const FateSpellDef *d = &gSpellDefs[i];
        bool known = fate_spell_known(i);
        bool selected = known && i == gFate.selectedSpell;
        put_gap();
        buf[0] = '\0';
        fate_strcat(buf, d->name);
        fate_strcat(buf, "  (");
        fate_strcat(buf, gSkillNames[d->school]);
        fate_strcat(buf, ", needs ");
        fate_strcat_int(buf, d->reqMag);
        fate_strcat(buf, " Magic)");
        put_text(buf, known ? (selected ? UI_HEADER : UI_TEXT) : UI_TEXT_DIM, LS_NORMAL);
        if (known) {
            put_button(selected ? "Selected" : "Select", ACT_SELECT_SPELL, i, selected);
        }
        end_line();
        buf[0] = '\0';
        fate_strcat(buf, d->desc);
        if (known) {
            fate_strcat(buf, "  Costs ");
            fate_strcat_int(buf, fate_spell_mana_cost(i));
            fate_strcat(buf, " mana.");
        } else {
            fate_strcat(buf, "  Sold by Cheato in Grove.");
        }
        put_line(buf, UI_TEXT_DIM, LS_SMALL);
    }
}

/* ------------------------------------------------------------------ */
/* Quests tab                                                          */
/* ------------------------------------------------------------------ */

static void put_quest(const FateQuest *q, s32 index, bool board) {
    char buf[256];
    put_gap();
    fate_quest_title(q, buf, sizeof(buf));
    put_line(buf, q->state == QSTATE_READY ? MSG_COLOR_XP : MSG_COLOR_QUEST, LS_NORMAL);
    fate_quest_describe(q, buf, sizeof(buf));
    put_line(buf, UI_TEXT, LS_SMALL);
    if (board) {
        put_button("Accept", ACT_QUEST_ACCEPT, index, TRUE);
    } else {
        if (q->state == QSTATE_READY && fate_in_town()) {
            put_button("Collect reward", ACT_QUEST_TURNIN, index, TRUE);
        }
        put_button("Abandon", ACT_QUEST_ABANDON, index, FALSE);
    }
    end_line();
}

static void build_quests(void) {
    s32 i;
    bool any = FALSE;

    use_col(0, 0.0f);
    header("Quest log");
    for (i = 0; i < MAX_ACTIVE_QUESTS; i++) {
        if (gFate.quests[i].type != QUEST_NONE) {
            put_quest(&gFate.quests[i], i, FALSE);
            any = TRUE;
        }
    }
    if (!any) {
        put_line("No quests. The folk of Grove post requests on the quest board.", UI_TEXT_DIM, LS_SMALL);
    }

    use_col(1, 0.0f);
    header("Grove quest board");
    if (!fate_in_town()) {
        put_line("Return to Grove (Spiral Mountain) to take on new quests and collect your rewards.", UI_TEXT_DIM, LS_SMALL);
        return;
    }
    any = FALSE;
    for (i = 0; i < QUEST_BOARD_SIZE; i++) {
        if (gFate.board[i].type != QUEST_NONE) {
            put_quest(&gFate.board[i], i, TRUE);
            any = TRUE;
        }
    }
    if (!any) {
        put_line("No new requests. Check back after your next adventure.", UI_TEXT_DIM, LS_SMALL);
    }
}

/* ------------------------------------------------------------------ */
/* Grove tab                                                           */
/* ------------------------------------------------------------------ */

static void build_grove(void) {
    char buf[160];
    s32 i;
    FateItem *sel = selected_item();

    use_col(0, 340.0f);
    header("Grove");
    for (i = 0; i < GROVE_COUNT; i++) {
        put_wide(sGroveNames[i], UI_TEXT, ACT_GROVE_SECTION, i, i == sGrove);
        end_line();
    }
    if (gFate.portalMap != 0) {
        put_gap();
        put_wide("Step through the portal", 0x9EE6FFFF, ACT_PORTAL, 0, FALSE);
        end_line();
    }
    put_gap();
    buf[0] = '\0';
    fate_strcat(buf, "Gold: ");
    fate_strcat_int(buf, (s32)gFate.gold);
    put_line(buf, MSG_COLOR_GOLD, LS_HEADER);

    use_col(1, 0.0f);
    header(sGroveNames[sGrove]);
    switch (sGrove) {
        case GROVE_BOTTLES: {
            bool any = FALSE;
            put_line("Bottles: \"Potions, scrolls, and I'll identify your finds, Banjo!\"", UI_TEXT_DIM, LS_SMALL);
            for (i = 0; i < CONS_COUNT; i++) {
                buf[0] = '\0';
                fate_strcat(buf, gConsumableNames[i]);
                fate_strcat(buf, "  (have ");
                fate_strcat_int(buf, gFate.consumables[i]);
                fate_strcat(buf, ")");
                put_price(buf, gConsumablePrices[i], ACT_BUY_CONS, i);
            }
            header("Sell");
            put_line("Sell single items from the Inventory tab while you're in Grove.", UI_TEXT_DIM, LS_SMALL);
            put_button("Sell all plain (white) items", ACT_SELL_JUNK, 0, FALSE);
            end_line();

            use_col(2, 520.0f);
            header("Identify");
            for (i = 0; i < INV_SIZE; i++) {
                if (!fate_item_empty(&gFate.inv[i]) && !fate_item_identified(&gFate.inv[i])) {
                    char name[96];
                    any = TRUE;
                    fate_item_name(&gFate.inv[i], name, sizeof(name));
                    put_text(name, fate_rarity_color(gFate.inv[i].rarity), LS_SMALL);
                    buf[0] = '\0';
                    fate_strcat(buf, "Identify (");
                    fate_strcat_int(buf, fate_identify_cost(&gFate.inv[i]));
                    fate_strcat(buf, "g)");
                    put_button(buf, ACT_IDENTIFY_PAID, i, FALSE);
                    end_line();
                }
            }
            if (!any) {
                put_line("Nothing to identify.", UI_TEXT_DIM, LS_SMALL);
            }
            break;
        }

        case GROVE_GOBI:
            put_line("Gobi: \"Lugged it all the way from the desert. New stock every visit.\"", UI_TEXT_DIM, LS_SMALL);
            for (i = 0; i < SHOP_STOCK; i++) {
                if (!fate_item_empty(&gShopStock[i])) {
                    char price[24];
                    put_item(&gShopStock[i], SEL_SHOP, i, NULL);
                    price[0] = '\0';
                    fate_strcat(price, "Buy (");
                    fate_strcat_int(price, fate_item_buy_price(&gShopStock[i]));
                    fate_strcat(price, "g)");
                    put_button(price, ACT_BUY_ITEM, i, gFate.gold >= (u32)fate_item_buy_price(&gShopStock[i]));
                    end_line();
                }
            }
            use_col(2, 520.0f);
            header("Details");
            put_item_details(sSelKind == SEL_SHOP ? sel : NULL);
            break;

        case GROVE_CHEATO: {
            bool any = FALSE;
            put_line("Cheato: \"Squawk! Knowledge isn't free, bear. Neither am I.\"", UI_TEXT_DIM, LS_SMALL);
            for (i = 0; i < SPELL_COUNT; i++) {
                if (!fate_spell_known(i)) {
                    any = TRUE;
                    put_gap();
                    buf[0] = '\0';
                    fate_strcat(buf, gSpellDefs[i].name);
                    fate_strcat(buf, "  (needs ");
                    fate_strcat_int(buf, gSpellDefs[i].reqMag);
                    fate_strcat(buf, " Magic)");
                    put_price(buf, gSpellDefs[i].price, ACT_BUY_SPELL, i);
                    put_line(gSpellDefs[i].desc, UI_TEXT_DIM, LS_SMALL);
                }
            }
            if (!any) {
                put_line("You know every spell Cheato sells.", MSG_COLOR_GOLD, LS_NORMAL);
            }
            break;
        }

        case GROVE_MUMBO:
            put_line("Mumbo: \"Mumbo add magic to item. Sometimes... it go wrong.\"", UI_TEXT_DIM, LS_SMALL);
            put_line("60% new or stronger magic, 25% nothing, 10% all magic lost, 5% curse.", UI_TEXT_DIM, LS_SMALL);
            for (i = 0; i < SLOT_COUNT; i++) {
                if (!fate_item_empty(&gFate.equip[i])) {
                    put_item(&gFate.equip[i], SEL_EQUIP, i, "(worn) ");
                    end_line();
                }
            }
            for (i = 0; i < INV_SIZE; i++) {
                if (!fate_item_empty(&gFate.inv[i])) {
                    put_item(&gFate.inv[i], SEL_INV, i, NULL);
                    end_line();
                }
            }
            use_col(2, 520.0f);
            header("Details");
            if ((sSelKind == SEL_INV || sSelKind == SEL_EQUIP) && sel != NULL && !fate_item_empty(sel)) {
                put_item_details(sel);
                put_gap();
                buf[0] = '\0';
                fate_strcat(buf, "Enchant (");
                fate_strcat_int(buf, fate_enchant_cost(sel));
                fate_strcat(buf, "g)");
                put_button(buf, ACT_ENCHANT, 0, TRUE);
                buf[0] = '\0';
                fate_strcat(buf, "Add socket (");
                fate_strcat_int(buf, fate_socket_cost(sel));
                fate_strcat(buf, "g)");
                put_button(buf, ACT_SOCKET_ADD, 0, FALSE);
                end_line();
            } else {
                put_item_details(NULL);
            }
            break;

        case GROVE_BRENTILDA:
            put_line("Brentilda: \"My nasty sister's minions treating you rough, dearie?", UI_TEXT_DIM, LS_SMALL);
            put_line("I'll patch you and your little friend up, free of charge.\"", UI_TEXT_DIM, LS_SMALL);
            put_gap();
            put_button("Heal Banjo, Kazooie and the pet", ACT_HEAL, 0, TRUE);
            end_line();
            break;

        case GROVE_TOOTY:
            put_line("Tooty: \"I'll tell everyone how brave my big brother is!\"", UI_TEXT_DIM, LS_SMALL);
            put_line("Fame unlocks Elite and Legendary gear, and every fame title grants 4 skill points.", UI_TEXT_DIM, LS_SMALL);
            put_gap();
            put_price("Spread tales of your deeds (+25 fame)", fate_fame_cost(), ACT_BUY_FAME, 0);
            break;

        case GROVE_BLUBBER:
            put_line("Captain Blubber: \"Har! Fish the waters where ye wade, and feed yer little friend", UI_TEXT_DIM, LS_SMALL);
            put_line("the catch. A flawless fish changes it for good!\"", UI_TEXT_DIM, LS_SMALL);
            put_gap();
            if (gFate.poleTier < 3) {
                static const char *poles[3] = { "Fishing Pole", "Sturdy Fishing Pole (bigger fish)", "Blubber's Golden Rod (best fish)" };
                put_price(poles[gFate.poleTier], fate_pole_price(), ACT_BUY_POLE, 0);
            } else {
                put_line("You own the Golden Rod.", MSG_COLOR_GOLD, LS_NORMAL);
            }
            header("Live fish (medium)");
            for (i = 0; i <= FISH_DOGFISH; i++) {
                if (i == FISH_BIGBUTT_BASS && fate_depth_for_level(LEVEL_3_CLANKERS_CAVERN) > 0 && gFate.questsCompleted < 3) {
                    continue; /* rare stock unlocks once the town trusts you */
                }
                put_price(gFishDefs[i].name, fate_fish_price(i, FISH_SIZE_MEDIUM) * 3, ACT_BUY_FISH, i);
            }
            put_gap();
            put_button("Sell all your fish", ACT_SELL_FISH, 0, FALSE);
            end_line();
            break;
    }
}

/* ------------------------------------------------------------------ */
/* Gruntilda's bargain                                                 */
/* ------------------------------------------------------------------ */

static void build_bargain(void) {
    char buf[160];

    use_col(0, 0.0f);
    put_line("Gruntilda appears in a puff of green smoke as you fall.", UI_TEXT, LS_NORMAL);
    put_line("\"Perhaps we can come to an... arrangement. Hee hee!\"", GRUNTY_GREEN, LS_NORMAL);
    put_gap();
    put_wide("Rise right here. I'll take some of your experience and fame.", UI_TEXT, ACT_BARGAIN, 0, TRUE);
    end_line();
    buf[0] = '\0';
    fate_strcat(buf, "Rise at this area's entrance. I'll take ");
    fate_strcat_int(buf, (s32)(gFate.gold / 10));
    fate_strcat(buf, " of your gold.");
    put_wide(buf, UI_TEXT, ACT_BARGAIN, 1, FALSE);
    end_line();
    buf[0] = '\0';
    fate_strcat(buf, "Flee to Grove and leave your ");
    fate_strcat_int(buf, (s32)gFate.gold);
    fate_strcat(buf, " gold behind (back within 5 minutes to reclaim it).");
    put_wide(buf, UI_TEXT, ACT_BARGAIN, 2, FALSE);
    end_line();
    put_wide("Refuse her. Accept your fate (a normal Banjo-Kazooie death).", UI_TEXT, ACT_BARGAIN, 3, FALSE);
    end_line();
}

static void build_page(void) {
    s32 c;

    page_begin();
    sTitle[0] = '\0';
    if (gFateBargainOpen) {
        fate_strcat(sTitle, "\"Not so fast, bear!\"");
        sTitleColor = GRUNTY_GREEN;
        build_bargain();
    } else {
        fate_strcat(sTitle, "Banjo and Kazooie  -  Level ");
        fate_strcat_int(sTitle, gFate.level);
        fate_strcat(sTitle, "  -  ");
        fate_strcat(sTitle, fate_fame_title());
        sTitleColor = UI_HEADER;
        switch (sTab) {
            case TAB_CHARACTER: build_character(); break;
            case TAB_INVENTORY: build_inventory(); break;
            case TAB_PET:       build_pet(); break;
            case TAB_SPELLS:    build_spells(); break;
            case TAB_QUESTS:    build_quests(); break;
            case TAB_GROVE:     build_grove(); break;
        }
    }

    for (c = 0; c < MENU_COLS; c++) {
        if (sSpecOverflow[c]) {
            LineSpec *l = &sSpec[c][COL_LINES - 1];
            copy_text(l->text, "... and more", LINE_TEXT, LINE_TEXT);
            l->color = UI_TEXT_DIM;
            l->style = LS_SMALL;
            l->buttons = 0;
        }
    }
}

/* ------------------------------------------------------------------ */
/* Pushing a page to the pool                                          */
/* ------------------------------------------------------------------ */

static void set_shown(RecompuiResource el, u8 *state, bool want, RecompuiDisplay display) {
    if (*state != want) {
        recompui_set_display(el, want ? display : DISPLAY_NONE);
        *state = want;
    }
}

static void button_look(RecompuiResource el, bool primary) {
    RecompuiColor c;
    ui_color(&c, primary ? BTN_BG_ON : BTN_BG);
    recompui_set_background_color(el, &c);
    ui_color(&c, primary ? BTN_EDGE_ON : BTN_EDGE);
    recompui_set_border_color(el, &c);
}

static void apply_button(PoolButton *p, const char *text, u32 color, bool primary, bool wide) {
    u32 h;
    set_shown(p->el, &p->shown, text != NULL, DISPLAY_FLEX);
    if (text == NULL) {
        return;
    }
    h = text_hash(text);
    if (h != p->hash) {
        recompui_set_text(p->label, text);
        p->hash = h;
    }
    if (color != p->color) {
        ui_set_text_color(p->label, color);
        p->color = color;
    }
    if (primary != p->primary) {
        button_look(p->el, primary);
        p->primary = primary;
    }
    if (wide != p->wide) {
        recompui_set_flex_grow(p->el, wide ? 1.0f : 0.0f);
        recompui_set_flex_shrink(p->el, wide ? 1.0f : 0.0f);
        recompui_set_justify_content(p->el, wide ? JUSTIFY_CONTENT_FLEX_START : JUSTIFY_CONTENT_CENTER);
        p->wide = wide;
    }
}

static void apply_line(PoolLine *p, const LineSpec *s) {
    bool hasText;
    s32 b;

    set_shown(p->row, &p->shown, s != NULL, DISPLAY_FLEX);
    if (s == NULL) {
        return;
    }
    if (s->gap != p->gap) {
        recompui_set_margin_top(p->row, s->gap ? 10.0f : 0.0f, UNIT_DP);
        p->gap = s->gap;
    }
    hasText = s->text[0] != '\0';
    set_shown(p->label, &p->labelShown, hasText, DISPLAY_BLOCK);
    if (hasText) {
        u32 h = text_hash(s->text);
        if (h != p->hash) {
            recompui_set_text(p->label, s->text);
            p->hash = h;
        }
        if (s->color != p->color) {
            ui_set_text_color(p->label, s->color);
            p->color = s->color;
        }
        if (s->style != p->style) {
            recompui_set_font_size(p->label, sStyleSize[s->style], UNIT_DP);
            recompui_set_line_height(p->label, sStyleSize[s->style] * 1.3f, UNIT_DP);
            recompui_set_font_weight(p->label, sStyleWeight[s->style]);
            p->style = s->style;
        }
    }
    for (b = 0; b < LINE_BUTTONS; b++) {
        if (b < s->buttons) {
            const ButtonSpec *bs = &s->btn[b];
            apply_button(&p->btn[b], bs->text, bs->color, bs->primary, bs->wide);
        } else {
            apply_button(&p->btn[b], NULL, 0, FALSE, FALSE);
        }
    }
}

static void apply_page(void) {
    s32 c, i;
    u32 h;
    bool showNotice = !gFateBargainOpen && sNotice[0] != '\0';

    h = text_hash(sTitle);
    if (h != sTitleHash) {
        recompui_set_text(sTitleLabel, sTitle);
        sTitleHash = h;
    }
    if (sTitleColor != sTitleShownColor) {
        ui_set_text_color(sTitleLabel, sTitleColor);
        sTitleShownColor = sTitleColor;
    }
    set_shown(sTabRow, &sTabRowShown, !gFateBargainOpen, DISPLAY_FLEX);
    apply_button(&sCloseBtn, gFateBargainOpen ? NULL : "Close", UI_TEXT, FALSE, FALSE);
    for (i = 0; i < TAB_COUNT; i++) {
        bool show = i != TAB_GROVE || fate_in_town();
        apply_button(&sTabBtn[i], show ? sTabNames[i] : NULL, i == sTab ? UI_HEADER : UI_TEXT, i == sTab, FALSE);
    }
    set_shown(sNoticeLabel, &sNoticeShown, showNotice, DISPLAY_BLOCK);
    if (showNotice) {
        h = text_hash(sNotice);
        if (h != sNoticeHash) {
            recompui_set_text(sNoticeLabel, sNotice);
            sNoticeHash = h;
        }
    }

    for (c = 0; c < MENU_COLS; c++) {
        PoolColumn *col = &sCols[c];
        f32 w = sSpecWidth[c];
        if (w != col->width) {
            if (w < 0.0f) {
                recompui_set_display(col->box, DISPLAY_NONE);
            } else {
                if (col->width < 0.0f) {
                    recompui_set_display(col->box, DISPLAY_FLEX);
                }
                recompui_set_flex_basis(col->box, w, UNIT_DP);
                recompui_set_flex_grow(col->box, w > 0.0f ? 0.0f : 1.0f);
                recompui_set_flex_shrink(col->box, w > 0.0f ? 0.0f : 1.0f);
            }
            col->width = w;
        }
        if (w < 0.0f) {
            continue;
        }
        for (i = 0; i < COL_LINES; i++) {
            apply_line(&col->lines[i], i < sSpecCount[c] ? &sSpec[c][i] : NULL);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Creating the pool                                                   */
/* ------------------------------------------------------------------ */

/*
 * Every property that apply_page changes later is set here first, so that
 * later changes overwrite an existing value instead of adding a new one.
 */
static void make_button(PoolButton *p, RecompuiResource parent, u32 clickId) {
    p->el = recompui_create_button(sCtx, parent, "", BUTTONSTYLE_SECONDARY);
    recompui_set_height(p->el, 36.0f, UNIT_DP);
    recompui_set_min_height(p->el, 36.0f, UNIT_DP);
    recompui_set_max_height(p->el, 36.0f, UNIT_DP);
    recompui_set_padding_left(p->el, 12.0f, UNIT_DP);
    recompui_set_padding_right(p->el, 12.0f, UNIT_DP);
    recompui_set_min_width(p->el, 0.0f, UNIT_DP);
    recompui_set_overflow(p->el, OVERFLOW_HIDDEN);
    recompui_set_flex_grow(p->el, 0.0f);
    recompui_set_flex_shrink(p->el, 0.0f);
    recompui_set_justify_content(p->el, JUSTIFY_CONTENT_CENTER);
    button_look(p->el, FALSE);
    recompui_set_display(p->el, DISPLAY_NONE);
    p->label = ui_label(sCtx, p->el, "", 17.0f, UI_TEXT);
    recompui_register_callback(p->el, on_click, (void *)clickId);
    p->hash = 0;
    p->color = UI_TEXT;
    p->shown = FALSE;
    p->primary = FALSE;
    p->wide = FALSE;
}

static void make_line(PoolLine *p, RecompuiResource parent, u32 firstClickId) {
    s32 b;
    p->row = ui_box(sCtx, parent, TRUE);
    recompui_set_gap(p->row, 8.0f, UNIT_DP);
    recompui_set_align_items(p->row, ALIGN_ITEMS_CENTER);
    recompui_set_margin_top(p->row, 0.0f, UNIT_DP);
    recompui_set_flex_shrink(p->row, 0.0f);
    recompui_set_display(p->row, DISPLAY_NONE);
    p->label = ui_label(sCtx, p->row, "", sStyleSize[LS_NORMAL], UI_TEXT);
    recompui_set_line_height(p->label, sStyleSize[LS_NORMAL] * 1.3f, UNIT_DP);
    recompui_set_font_weight(p->label, sStyleWeight[LS_NORMAL]);
    recompui_set_flex_grow(p->label, 1.0f);
    recompui_set_flex_shrink(p->label, 1.0f);
    recompui_set_display(p->label, DISPLAY_BLOCK);
    p->hash = 0;
    p->color = UI_TEXT;
    p->style = LS_NORMAL;
    p->gap = 0;
    p->shown = FALSE;
    p->labelShown = TRUE;
    for (b = 0; b < LINE_BUTTONS; b++) {
        make_button(&p->btn[b], p->row, firstClickId + b);
    }
}

static void build_pool(void) {
    RecompuiResource root, panel, top, body;
    s32 c, i;

    sCtx = recompui_create_context();
    recompui_open_context(sCtx);
    root = recompui_context_root(sCtx);
    ui_absolute(root, 0.0f, 0.0f, 0.0f, 0.0f);
    recompui_set_width(root, 100.0f, UNIT_PERCENT);
    recompui_set_height(root, 100.0f, UNIT_PERCENT);
    recompui_set_display(root, DISPLAY_FLEX);
    recompui_set_justify_content(root, JUSTIFY_CONTENT_CENTER);
    recompui_set_align_items(root, ALIGN_ITEMS_CENTER);
    ui_set_bg(root, 0x00000099);

    panel = ui_box(sCtx, root, FALSE);
    recompui_set_width(panel, 1500.0f, UNIT_DP);
    recompui_set_height(panel, 940.0f, UNIT_DP);
    recompui_set_max_width(panel, 96.0f, UNIT_PERCENT);
    recompui_set_gap(panel, 10.0f, UNIT_DP);
    ui_set_bg(panel, UI_PANEL_BG);
    ui_set_border(panel, UI_PANEL_EDGE, 3.0f, 16.0f);
    recompui_set_padding(panel, 20.0f, UNIT_DP);

    top = ui_box(sCtx, panel, TRUE);
    recompui_set_align_items(top, ALIGN_ITEMS_CENTER);
    recompui_set_justify_content(top, JUSTIFY_CONTENT_SPACE_BETWEEN);
    sTitleLabel = ui_label(sCtx, top, "", 28.0f, UI_HEADER);
    recompui_set_font_weight(sTitleLabel, 700);
    make_button(&sCloseBtn, top, CLICK_CLOSE);

    sTabRow = ui_box(sCtx, panel, TRUE);
    recompui_set_gap(sTabRow, 8.0f, UNIT_DP);
    for (i = 0; i < TAB_COUNT; i++) {
        make_button(&sTabBtn[i], sTabRow, CLICK_TAB + i);
    }

    sNoticeLabel = ui_label(sCtx, panel, "", 18.0f, MSG_COLOR_QUEST);
    recompui_set_display(sNoticeLabel, DISPLAY_NONE);

    body = ui_box(sCtx, panel, TRUE);
    recompui_set_gap(body, 10.0f, UNIT_DP);
    recompui_set_height(body, 790.0f, UNIT_DP);
    recompui_set_flex_shrink(body, 0.0f);
    for (c = 0; c < MENU_COLS; c++) {
        PoolColumn *col = &sCols[c];
        col->box = ui_box(sCtx, body, FALSE);
        ui_set_bg(col->box, UI_INSET_BG);
        ui_set_border(col->box, 0x6B5530FF, 1.0f, 8.0f);
        recompui_set_padding(col->box, 10.0f, UNIT_DP);
        recompui_set_gap(col->box, 4.0f, UNIT_DP);
        recompui_set_height(col->box, 100.0f, UNIT_PERCENT);
        recompui_set_min_width(col->box, 0.0f, UNIT_DP);
        recompui_set_overflow_y(col->box, OVERFLOW_AUTO);
        recompui_set_flex_basis(col->box, 0.0f, UNIT_DP);
        recompui_set_flex_grow(col->box, 1.0f);
        recompui_set_flex_shrink(col->box, 1.0f);
        recompui_set_display(col->box, DISPLAY_FLEX);
        col->width = 0.0f;
        for (i = 0; i < COL_LINES; i++) {
            make_line(&col->lines[i], col->box, (c * COL_LINES + i) * LINE_BUTTONS);
        }
    }

    sTitleHash = 0;
    sTitleShownColor = UI_HEADER;
    sNoticeHash = 0;
    sTabRowShown = TRUE;
    sNoticeShown = FALSE;
    recompui_close_context(sCtx);
    sPoolBuilt = TRUE;
}

/* ------------------------------------------------------------------ */
/* Show / hide                                                         */
/* ------------------------------------------------------------------ */

void fate_menu_init(void) {
    /* The pool is built the first time the menu opens, while nothing else is on screen. */
}

static void show(void) {
    if (!sShown) {
        if (!sPoolBuilt) {
            /* Hiding the HUD waits for the render thread to finish its frame. With no
               context shown it skips RmlUi's update, so the pool can be built safely. */
            fate_hud_hide_now();
            build_pool();
        }
        sShown = TRUE;
        gFateMenuOpen = TRUE;
        sWorldWasEnabled = func_8033511C();
        func_80335110(0); /* freeze the world, exactly like the pause menu */
        build_page();
        recompui_open_context(sCtx);
        apply_page();
        recompui_close_context(sCtx);
        recompui_show_context(sCtx);
        fate_play_sfx(SFX_C5_TWINKLY_POP);
    }
    sRebuild = TRUE;
}

void fate_menu_open(s32 tab) {
    if (!fate_gameplay_active()) {
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
    if (!sPoolBuilt || !sShown) {
        return;
    }
    if (sRebuild) {
        sRebuild = FALSE;
        build_page();
        recompui_open_context(sCtx);
        apply_page();
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
    u32 id = (u32)userdata;
    s32 c, line, b;
    const LineSpec *l;

    if (event->type != UI_EVENT_CLICK || !sShown) {
        return;
    }
    if (id == CLICK_CLOSE) {
        if (!gFateBargainOpen) {
            fate_menu_close();
        }
        return;
    }
    if (id >= CLICK_TAB && id < CLICK_TAB + TAB_COUNT) {
        sTab = id - CLICK_TAB;
        sSelKind = SEL_NONE;
        sNotice[0] = '\0';
        sRebuild = TRUE;
        return;
    }
    b = id % LINE_BUTTONS;
    line = (id / LINE_BUTTONS) % COL_LINES;
    c = id / (LINE_BUTTONS * COL_LINES);
    if (c >= MENU_COLS || line >= sSpecCount[c]) {
        return;
    }
    l = &sSpec[c][line];
    if (b >= l->buttons) {
        return;
    }
    do_action(l->btn[b].action, l->btn[b].arg);
}
