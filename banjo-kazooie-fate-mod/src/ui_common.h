#ifndef FATE_UI_COMMON_H
#define FATE_UI_COMMON_H

#include "fate.h"
#include "recompui.h"

/* Palette: parchment and dark wood, Fate's menus with Banjo-Kazooie colors. */
#define UI_PANEL_BG    0x1B1410E8
#define UI_PANEL_EDGE  0xC9973AFF
#define UI_INSET_BG    0x2A2019F0
#define UI_TEXT        0xF1E7D0FF
#define UI_TEXT_DIM    0xB9AA8CFF
#define UI_HEADER      0xF2C744FF
#define UI_HP_FILL     0xD8402FFF
#define UI_HP_BACK     0x4A1410FF
#define UI_MANA_FILL   0x3F7BE0FF
#define UI_MANA_BACK   0x13244AFF
#define UI_XP_FILL     0xE8C14BFF
#define UI_XP_BACK     0x3A3010FF
#define UI_PET_FILL    0x5FC95AFF
#define UI_PET_BACK    0x173A15FF

void ui_color(RecompuiColor *out, u32 rgba);
void ui_set_bg(RecompuiResource el, u32 rgba);
void ui_set_text_color(RecompuiResource el, u32 rgba);
void ui_set_border(RecompuiResource el, u32 rgba, f32 width, f32 radius);
RecompuiResource ui_box(RecompuiContext ctx, RecompuiResource parent, bool row);
RecompuiResource ui_label(RecompuiContext ctx, RecompuiResource parent, const char *text, f32 size, u32 color);
RecompuiResource ui_bar(RecompuiContext ctx, RecompuiResource parent, f32 width, f32 height, u32 back, u32 fill, RecompuiResource *outFill);
void ui_bar_set(RecompuiResource fill, f32 fraction);
void ui_absolute(RecompuiResource el, f32 left, f32 top, f32 right, f32 bottom);

#define UI_UNSET (-99999.0f)

#endif
