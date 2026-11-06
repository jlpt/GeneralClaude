#pragma once
#include "raylib.h"

// A horizontal slider dragged with the left mouse button.
class Slider {
public:
    Slider(int x, int y, int width, int minValue, int maxValue, int value)
        : x_(x), y_(y), width_(width), min_(minValue), max_(maxValue), value_(value) {}

    // Returns true when the value changed.
    bool Update() {
        if (!IsMouseButtonDown(MOUSE_BUTTON_LEFT) || !Contains(GetMousePosition())) return false;
        const float t = (GetMousePosition().x - float(x_)) / float(width_);
        int value = min_ + int(t * float(max_ - min_));
        value = value < min_ ? min_ : (value > max_ ? max_ : value);
        const bool changed = value != value_;
        value_ = value;
        return changed;
    }

    bool Contains(Vector2 point) const {
        return point.x >= float(x_) && point.x <= float(x_ + width_) && point.y >= float(y_ - 12) &&
               point.y <= float(y_ + 12);
    }

    void Draw(const char* label) const {
        DrawRectangle(x_, y_ - 1, width_, 3, DARKGRAY);
        const float t = float(value_ - min_) / float(max_ - min_);
        DrawCircle(x_ + int(t * float(width_)), y_, 8, RED);
        DrawText(label, x_ + width_ + 16, y_ - 8, 16, WHITE);
    }

    int Value() const { return value_; }

private:
    int x_, y_, width_;
    int min_, max_;
    int value_;
};
