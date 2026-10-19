#include "ui_common.h"

/*
 * Fate-style HUD drawn with recompui on top of the game:
 *   bottom center  health and mana bars, level, gold, depth, potions, spell, xp bar
 *   bottom left    pet panel
 *   upper center   target bar for the last monster hit (gold for champions)
 *   left side      scrolling loot and event log
 *   center         banners (level up, quest complete) and the fishing prompt
 */

#define LOG_LINES 6
#define LOG_SECONDS 7.0f

extern char gFateTargetName[64];
extern s32 gFateTargetHp;
extern s32 gFateTargetMaxHp;
extern u8 gFateTargetChampion;

static RecompuiContext sCtx = RECOMPUI_NULL_CONTEXT;
static bool sShown;
static bool sDirty = TRUE;
static f32 sRefreshTimer;

static RecompuiResource sHpFill, sHpText, sManaFill, sManaText, sXpFill;
static RecompuiResource sLevelText, sGoldText, sDepthText, sBeltText, sSpellText;
static RecompuiResource sPetPanel, sPetName, sPetFill, sPetStatus;
static RecompuiResource sTargetPanel, sTargetName, sTargetFill, sTargetDmg;
static RecompuiResource sBannerText, sPromptText;
static RecompuiResource sLogText[LOG_LINES];

static struct {
    char text[LOG_LINES][112];
    u32 color[LOG_LINES];
    f32 age[LOG_LINES];
    s32 count;
} sLog;

static char sBanner[64];
static u32 sBannerColor;
static f32 sBannerTimer;

/* Cache of what's on screen so recompui only gets calls when something changes. */
#define TEXT_CACHE_SLOTS 24
static char sShownText[TEXT_CACHE_SLOTS][112];

static void set_text_cached(s32 slot, RecompuiResource el, const char *text) {
    s32 i;
    for (i = 0; i < 111 && text[i] == sShownText[slot][i] && text[i] != '\0'; i++) {
    }
    if (text[i] == sShownText[slot][i]) {
        return;
    }
    fate_strcpy(sShownText[slot], text);
    recompui_set_text(el, text);
}

void fate_hud_mark_dirty(void) {
    sDirty = TRUE;
}

void fate_message(const char *text, u32 color) {
    s32 i;
    /* newest at index 0 */
    for (i = LOG_LINES - 1; i > 0; i--) {
        fate_strcpy(sLog.text[i], sLog.text[i - 1]);
        sLog.color[i] = sLog.color[i - 1];
        sLog.age[i] = sLog.age[i - 1];
    }
    i = 0;
    while (text[i] && i < 111) {
        sLog.text[0][i] = text[i];
        i++;
    }
    sLog.text[0][i] = '\0';
    sLog.color[0] = color;
    sLog.age[0] = 0.0f;
    if (sLog.count < LOG_LINES) {
        sLog.count++;
    }
    recomp_printf("[Fate] %s\n", text);
    sDirty = TRUE;
}

void fate_banner(const char *text, u32 color) {
    s32 i = 0;
    while (text[i] && i < 63) {
        sBanner[i] = text[i];
        i++;
    }
    sBanner[i] = '\0';
    sBannerColor = color;
    sBannerTimer = 3.0f;
    sDirty = TRUE;
}

/* ------------------------------------------------------------------ */
/* Layout                                                              */
/* ------------------------------------------------------------------ */

static RecompuiResource panel(RecompuiResource parent, bool row) {
    RecompuiResource p = ui_box(sCtx, parent, row);
    ui_set_bg(p, UI_PANEL_BG);
    ui_set_border(p, UI_PANEL_EDGE, 2.0f, 12.0f);
    recompui_set_padding(p, 12.0f, UNIT_DP);
    return p;
}

void fate_hud_init(void) {
    RecompuiResource root, bottom, left, mid, right, row, xp, logBox;
    s32 i;

    sCtx = recompui_create_context();
    recompui_set_context_captures_input(sCtx, 0);
    recompui_set_context_captures_mouse(sCtx, 0);
    recompui_open_context(sCtx);
    root = recompui_context_root(sCtx);
    ui_absolute(root, 0.0f, 0.0f, 0.0f, 0.0f);
    recompui_set_width(root, 100.0f, UNIT_PERCENT);
    recompui_set_height(root, 100.0f, UNIT_PERCENT);

    /* Bottom bar */
    bottom = panel(root, TRUE);
    ui_absolute(bottom, UI_UNSET, UI_UNSET, UI_UNSET, 18.0f);
    recompui_set_left(bottom, 50.0f, UNIT_PERCENT);
    recompui_set_margin_left(bottom, -430.0f, UNIT_DP);
    recompui_set_width(bottom, 860.0f, UNIT_DP);
    recompui_set_flex_direction(bottom, FLEX_DIRECTION_COLUMN);
    recompui_set_gap(bottom, 6.0f, UNIT_DP);

    row = ui_box(sCtx, bottom, TRUE);
    recompui_set_gap(row, 14.0f, UNIT_DP);
    recompui_set_align_items(row, ALIGN_ITEMS_CENTER);

    left = ui_box(sCtx, row, FALSE);
    recompui_set_gap(left, 2.0f, UNIT_DP);
    sHpText = ui_label(sCtx, left, "Health", 20.0f, UI_TEXT);
    ui_bar(sCtx, left, 270.0f, 22.0f, UI_HP_BACK, UI_HP_FILL, &sHpFill);

    mid = ui_box(sCtx, row, FALSE);
    recompui_set_flex_grow(mid, 1.0f);
    recompui_set_align_items(mid, ALIGN_ITEMS_CENTER);
    sLevelText = ui_label(sCtx, mid, "", 22.0f, UI_HEADER);
    sGoldText = ui_label(sCtx, mid, "", 18.0f, UI_TEXT);
    sDepthText = ui_label(sCtx, mid, "", 16.0f, UI_TEXT_DIM);

    right = ui_box(sCtx, row, FALSE);
    recompui_set_gap(right, 2.0f, UNIT_DP);
    recompui_set_align_items(right, ALIGN_ITEMS_FLEX_END);
    sManaText = ui_label(sCtx, right, "Mana", 20.0f, UI_TEXT);
    ui_bar(sCtx, right, 270.0f, 22.0f, UI_MANA_BACK, UI_MANA_FILL, &sManaFill);

    row = ui_box(sCtx, bottom, TRUE);
    recompui_set_justify_content(row, JUSTIFY_CONTENT_SPACE_BETWEEN);
    sBeltText = ui_label(sCtx, row, "", 16.0f, UI_TEXT_DIM);
    sSpellText = ui_label(sCtx, row, "", 16.0f, 0xE58CF2FF);

    xp = ui_bar(sCtx, bottom, 836.0f, 8.0f, UI_XP_BACK, UI_XP_FILL, &sXpFill);
    (void)xp;

    /* Pet panel */
    sPetPanel = panel(root, FALSE);
    ui_absolute(sPetPanel, 24.0f, UI_UNSET, UI_UNSET, 18.0f);
    recompui_set_width(sPetPanel, 300.0f, UNIT_DP);
    recompui_set_gap(sPetPanel, 4.0f, UNIT_DP);
    sPetName = ui_label(sCtx, sPetPanel, "", 18.0f, 0x7FD8FFFF);
    ui_bar(sCtx, sPetPanel, 276.0f, 14.0f, UI_PET_BACK, UI_PET_FILL, &sPetFill);
    sPetStatus = ui_label(sCtx, sPetPanel, "", 15.0f, UI_TEXT_DIM);

    /* Target bar */
    sTargetPanel = panel(root, FALSE);
    ui_absolute(sTargetPanel, UI_UNSET, 150.0f, UI_UNSET, UI_UNSET);
    recompui_set_left(sTargetPanel, 50.0f, UNIT_PERCENT);
    recompui_set_margin_left(sTargetPanel, -230.0f, UNIT_DP);
    recompui_set_width(sTargetPanel, 460.0f, UNIT_DP);
    recompui_set_align_items(sTargetPanel, ALIGN_ITEMS_CENTER);
    recompui_set_gap(sTargetPanel, 4.0f, UNIT_DP);
    sTargetName = ui_label(sCtx, sTargetPanel, "", 20.0f, UI_TEXT);
    ui_bar(sCtx, sTargetPanel, 430.0f, 16.0f, UI_HP_BACK, UI_HP_FILL, &sTargetFill);
    sTargetDmg = ui_label(sCtx, sTargetPanel, "", 16.0f, 0xFFD27FFF);
    recompui_set_display(sTargetPanel, DISPLAY_NONE);

    /* Banner and prompt */
    sBannerText = ui_label(sCtx, root, "", 54.0f, UI_HEADER);
    ui_absolute(sBannerText, 0.0f, 300.0f, 0.0f, UI_UNSET);
    recompui_set_text_align(sBannerText, TEXT_ALIGN_CENTER);
    recompui_set_font_weight(sBannerText, 700);

    sPromptText = ui_label(sCtx, root, "", 26.0f, 0x9EE6FFFF);
    ui_absolute(sPromptText, 0.0f, UI_UNSET, 0.0f, 205.0f);
    recompui_set_text_align(sPromptText, TEXT_ALIGN_CENTER);

    /* Event log */
    logBox = ui_box(sCtx, root, FALSE);
    ui_absolute(logBox, 24.0f, 260.0f, UI_UNSET, UI_UNSET);
    recompui_set_width(logBox, 620.0f, UNIT_DP);
    recompui_set_gap(logBox, 3.0f, UNIT_DP);
    for (i = 0; i < LOG_LINES; i++) {
        sLogText[i] = ui_label(sCtx, logBox, "", 17.0f, UI_TEXT);
    }

    recompui_close_context(sCtx);
    for (i = 0; i < TEXT_CACHE_SLOTS; i++) {
        sShownText[i][0] = '\x01';
        sShownText[i][1] = '\0';
    }
}

/* ------------------------------------------------------------------ */
/* Refresh                                                             */
/* ------------------------------------------------------------------ */

static void refresh(void) {
    char buf[112];
    s32 i;
    u32 xpBase = fate_xp_for_level(gFate.level);
    u32 xpNext = fate_xp_for_level(gFate.level + 1);
    f32 xpFrac = (gFate.level >= FATE_MAX_LEVEL) ? 1.0f : (f32)(gFate.xp - xpBase) / (f32)(xpNext - xpBase);

    buf[0] = '\0';
    fate_strcat(buf, "Health ");
    fate_strcat_int(buf, (s32)(gFateHp + 0.5f));
    fate_strcat(buf, " / ");
    fate_strcat_int(buf, gDerived.maxHp);
    set_text_cached(0, sHpText, buf);
    ui_bar_set(sHpFill, gDerived.maxHp > 0 ? gFateHp / (f32)gDerived.maxHp : 0.0f);

    buf[0] = '\0';
    fate_strcat(buf, "Mana ");
    fate_strcat_int(buf, (s32)gFateMana);
    fate_strcat(buf, " / ");
    fate_strcat_int(buf, gDerived.maxMana);
    set_text_cached(1, sManaText, buf);
    ui_bar_set(sManaFill, gDerived.maxMana > 0 ? gFateMana / (f32)gDerived.maxMana : 0.0f);
    ui_bar_set(sXpFill, xpFrac);

    buf[0] = '\0';
    fate_strcat(buf, "Level ");
    fate_strcat_int(buf, gFate.level);
    if (gFate.statPoints || gFate.skillPoints) {
        fate_strcat(buf, "  (points to spend!)");
    }
    set_text_cached(2, sLevelText, buf);

    buf[0] = '\0';
    fate_strcat_int(buf, (s32)gFate.gold);
    fate_strcat(buf, " gold   |   ");
    fate_strcat(buf, fate_fame_title());
    set_text_cached(3, sGoldText, buf);

    buf[0] = '\0';
    if (fate_in_town()) {
        fate_strcat(buf, "Grove - safe haven");
    } else {
        fate_strcat(buf, "Depth ");
        fate_strcat_int(buf, fate_depth());
        fate_strcat(buf, "  -  ");
        fate_strcat(buf, fate_level_name(level_get()));
    }
    set_text_cached(4, sDepthText, buf);

    buf[0] = '\0';
    fate_strcat(buf, "Health potions ");
    fate_strcat_int(buf, gFate.consumables[CONS_HP_POTION] + gFate.consumables[CONS_HP_POTION_GREATER]);
    fate_strcat(buf, " (Down)   Mana ");
    fate_strcat_int(buf, gFate.consumables[CONS_MANA_POTION] + gFate.consumables[CONS_MANA_POTION_GREATER]);
    fate_strcat(buf, " (Left)   Return ");
    fate_strcat_int(buf, gFate.consumables[CONS_TOWN_PORTAL]);
    set_text_cached(5, sBeltText, buf);

    buf[0] = '\0';
    if (fate_spell_known(gFate.selectedSpell)) {
        fate_strcat(buf, gSpellDefs[gFate.selectedSpell].name);
        fate_strcat(buf, " (");
        fate_strcat_int(buf, fate_spell_mana_cost(gFate.selectedSpell));
        fate_strcat(buf, " mana)");
        if (fate_spell_cooldown(gFate.selectedSpell) > 0.0f) {
            fate_strcat(buf, " ...");
        }
        fate_strcat(buf, "  Right");
    }
    set_text_cached(6, sSpellText, buf);

    /* pet */
    if (gFate.petSpecies == PET_SPECIES_NONE) {
        recompui_set_display(sPetPanel, DISPLAY_NONE);
    } else {
        recompui_set_display(sPetPanel, DISPLAY_FLEX);
        buf[0] = '\0';
        fate_strcat(buf, fate_pet_name());
        fate_strcat(buf, "  Lv ");
        fate_strcat_int(buf, gFate.petLevel);
        set_text_cached(7, sPetName, buf);
        ui_bar_set(sPetFill, (f32)gFate.petHp / (f32)fate_maxi(1, fate_pet_max_hp()));
        buf[0] = '\0';
        fate_strcat(buf, fate_pet_status());
        if (fate_pet_town_seconds() > 0.0f) {
            fate_strcat(buf, " (");
            fate_strcat_int(buf, (s32)fate_pet_town_seconds());
            fate_strcat(buf, "s)");
        } else if (gFate.petFormSeconds > 0) {
            fate_strcat(buf, "  - form ");
            fate_strcat_int(buf, gFate.petFormSeconds);
            fate_strcat(buf, "s");
        }
        set_text_cached(8, sPetStatus, buf);
    }

    /* target */
    if (gFateTargetTimer > 0.0f && gFateTargetMaxHp > 0) {
        recompui_set_display(sTargetPanel, DISPLAY_FLEX);
        set_text_cached(9, sTargetName, gFateTargetName);
        ui_set_text_color(sTargetName, gFateTargetChampion ? MSG_COLOR_GOLD : UI_TEXT);
        ui_bar_set(sTargetFill, (f32)gFateTargetHp / (f32)gFateTargetMaxHp);
        buf[0] = '\0';
        fate_strcat(buf, "-");
        fate_strcat_int(buf, gFateLastDamage);
        if (gFateLastCrit) {
            fate_strcat(buf, "  CRITICAL!");
        }
        if (gFateTargetChampion) {
            fate_strcat(buf, "   [");
            fate_strcat(buf, gChampionModNames[gFateTargetChampion]);
            fate_strcat(buf, " Champion]");
        }
        set_text_cached(10, sTargetDmg, buf);
    } else {
        recompui_set_display(sTargetPanel, DISPLAY_NONE);
    }

    set_text_cached(11, sBannerText, sBannerTimer > 0.0f ? sBanner : "");
    ui_set_text_color(sBannerText, sBannerColor);
    set_text_cached(12, sPromptText, fate_fishing_prompt());

    for (i = 0; i < LOG_LINES; i++) {
        bool visible = i < sLog.count && sLog.age[i] < LOG_SECONDS;
        set_text_cached(13 + i, sLogText[i], visible ? sLog.text[i] : "");
        if (visible) {
            ui_set_text_color(sLogText[i], sLog.color[i]);
            recompui_set_opacity(sLogText[i], sLog.age[i] > LOG_SECONDS - 1.5f ? (LOG_SECONDS - sLog.age[i]) / 1.5f : 1.0f);
        }
    }
}

void fate_hud_frame(void) {
    f32 dt = time_getDelta();
    bool want;
    s32 i;

    if (sCtx == RECOMPUI_NULL_CONTEXT) {
        return;
    }
    if (dt < 0.0f || dt > 0.25f) {
        dt = 0.0f;
    }
    want = fate_gameplay_active() && !gFateMenuOpen && recomp_get_config_u32("show_hud") == 0;
    if (want != sShown) {
        sShown = want;
        if (want) {
            recompui_show_context(sCtx);
            sDirty = TRUE;
        } else {
            recompui_hide_context(sCtx);
        }
    }
    if (!sShown) {
        return;
    }

    for (i = 0; i < sLog.count; i++) {
        sLog.age[i] += dt;
    }
    if (sBannerTimer > 0.0f) {
        sBannerTimer -= dt;
        if (sBannerTimer <= 0.0f) {
            sDirty = TRUE;
        }
    }

    /* Bars and the log fade animate, so refresh ten times a second at most. */
    sRefreshTimer -= dt;
    if (!sDirty && sRefreshTimer > 0.0f) {
        return;
    }
    sRefreshTimer = 0.1f;
    sDirty = FALSE;
    recompui_open_context(sCtx);
    refresh();
    recompui_close_context(sCtx);
}
