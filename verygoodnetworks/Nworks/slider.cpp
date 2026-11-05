#include "raylib.h"

class Slider {
public:
    int x, y, width;
    int minValue, maxValue;
    int currentValue;

    Slider(int posX, int posY, int w, int minVal, int maxVal, int startVal)
        : x(posX), y(posY), width(w), minValue(minVal), maxValue(maxVal), currentValue(startVal) {}

    void Update() {
        Vector2 mouse = GetMousePosition();

        // If mouse is clicked and hovering the slider line (you can adjust Y range for easier clicking)
        if (IsMouseButtonDown(MOUSE_LEFT_BUTTON) && mouse.x >= x && mouse.x <= (x + width) && mouse.y >= y - 20 && mouse.y <= y + 20) {
            // Calculate new value based on mouse.x position
            float percentage = (mouse.x - x) / width;
            currentValue = static_cast<int>(minValue + percentage * (maxValue - minValue));

            // Clamp between min and max
            if (currentValue < minValue) currentValue = minValue;
            if (currentValue > maxValue) currentValue = maxValue;

            // Set the FPS target in raylib
            SetTargetFPS(currentValue);
        }
    }

    void Draw() {
        // Draw slider line
        DrawLine(x, y, x + width, y, DARKGRAY);
        DrawLine(x, y+1, x + width, y+1, DARKGRAY);
        DrawLine(x, y-1, x + width, y-1, DARKGRAY);

        // Calculate handle position
        float percentage = static_cast<float>(currentValue - minValue) / (maxValue - minValue);
        int handleX = static_cast<int>(x + percentage * width);

        // Draw handle
        DrawCircle(handleX, y, 8, RED);

        // Draw current FPS value above the slider
       // DrawText(TextFormat("Target FPS: %d", currentValue), x, y - 30, 20, WHITE);
    }
};
