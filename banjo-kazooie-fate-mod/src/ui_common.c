#include "ui_common.h"

void ui_color(RecompuiColor *out, u32 rgba) {
    out->r = (rgba >> 24) & 0xFF;
    out->g = (rgba >> 16) & 0xFF;
    out->b = (rgba >> 8) & 0xFF;
    out->a = rgba & 0xFF;
}

void ui_set_bg(RecompuiResource el, u32 rgba) {
    RecompuiColor c;
    ui_color(&c, rgba);
    recompui_set_background_color(el, &c);
}

void ui_set_text_color(RecompuiResource el, u32 rgba) {
    RecompuiColor c;
    ui_color(&c, rgba);
    recompui_set_color(el, &c);
}

void ui_set_border(RecompuiResource el, u32 rgba, f32 width, f32 radius) {
    RecompuiColor c;
    ui_color(&c, rgba);
    recompui_set_border_width(el, width, UNIT_DP);
    recompui_set_border_color(el, &c);
    recompui_set_border_radius(el, radius, UNIT_DP);
}

RecompuiResource ui_box(RecompuiContext ctx, RecompuiResource parent, bool row) {
    RecompuiResource el = recompui_create_element(ctx, parent);
    recompui_set_display(el, DISPLAY_FLEX);
    recompui_set_flex_direction(el, row ? FLEX_DIRECTION_ROW : FLEX_DIRECTION_COLUMN);
    return el;
}

RecompuiResource ui_label(RecompuiContext ctx, RecompuiResource parent, const char *text, f32 size, u32 color) {
    RecompuiResource el = recompui_create_label(ctx, parent, text, LABELSTYLE_NORMAL);
    recompui_set_font_size(el, size, UNIT_DP);
    recompui_set_line_height(el, size * 1.25f, UNIT_DP);
    ui_set_text_color(el, color);
    return el;
}

RecompuiResource ui_bar(RecompuiContext ctx, RecompuiResource parent, f32 width, f32 height, u32 back, u32 fill, RecompuiResource *outFill) {
    RecompuiResource bar = recompui_create_element(ctx, parent);
    RecompuiResource inner;
    recompui_set_width(bar, width, UNIT_DP);
    recompui_set_height(bar, height, UNIT_DP);
    ui_set_bg(bar, back);
    ui_set_border(bar, 0x00000099, 2.0f, height / 2.0f);
    recompui_set_overflow(bar, OVERFLOW_HIDDEN);
    inner = recompui_create_element(ctx, bar);
    recompui_set_height(inner, 100.0f, UNIT_PERCENT);
    recompui_set_width(inner, 100.0f, UNIT_PERCENT);
    ui_set_bg(inner, fill);
    recompui_set_border_radius(inner, height / 2.0f, UNIT_DP);
    *outFill = inner;
    return bar;
}

void ui_bar_set(RecompuiResource fill, f32 fraction) {
    recompui_set_width(fill, fate_clampf(fraction, 0.0f, 1.0f) * 100.0f, UNIT_PERCENT);
}

void ui_absolute(RecompuiResource el, f32 left, f32 top, f32 right, f32 bottom) {
    recompui_set_position(el, POSITION_ABSOLUTE);
    if (left != UI_UNSET) recompui_set_left(el, left, UNIT_DP);
    if (top != UI_UNSET) recompui_set_top(el, top, UNIT_DP);
    if (right != UI_UNSET) recompui_set_right(el, right, UNIT_DP);
    if (bottom != UI_UNSET) recompui_set_bottom(el, bottom, UNIT_DP);
}
