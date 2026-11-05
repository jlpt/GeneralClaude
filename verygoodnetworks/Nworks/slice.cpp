#include "raylib.h"
#include "slice.h"

Slice::Slice(float x1, float y1, float x2, float y2)
    : x1(x1), y1(y1), x2(x2), y2(y2) {}

bool Slice::IsActive() const {
    return frame >= fadeInDuration && frame < fadeInDuration + activeDuration;
}

int Slice::GetOpacity() const {
    if (frame < fadeInDuration) {
        return static_cast<int>((frame / static_cast<float>(fadeInDuration)) * 255);
    }
    else if (frame < fadeInDuration + activeDuration) {
        return 255;
    }
    else if (frame < fadeInDuration + activeDuration + fadeOutDuration) {
        float progress = (frame - fadeInDuration - activeDuration) / static_cast<float>(fadeOutDuration);
        return static_cast<int>((1.0f - progress) * 255);
    }
    return 0;
}

void Slice::Draw() {
    int opacity = GetOpacity();
    if (opacity <= 0) return;

    Color inner = Color{ 0, 0, 0, static_cast<unsigned char>(opacity) };
    Color outer = Color{ 255, 255, 255, static_cast<unsigned char>(opacity) };

    DrawLineEx({ x1, y1 }, { x2, y2 }, static_cast<float>(width), outer);
    DrawLineEx({ x1, y1 }, { x2, y2 }, static_cast<float>(width - 1), inner);
}

void Slice::Update() {
    frame++;
    if (frame > fadeInDuration + activeDuration + fadeOutDuration) {
        isDead = true;
    }
}

// Distance from point to line segment
bool Slice::CollidesWith(float x, float y, float radius) const {
    float vx = x2 - x1;
    float vy = y2 - y1;
    float wx = x - x1;
    float wy = y - y1;

    float lenSquared = vx * vx + vy * vy;
    float t = (vx * wx + vy * wy) / lenSquared;
    t = std::fmax(0, std::fmin(1, t));

    float closestX = x1 + t * vx;
    float closestY = y1 + t * vy;

    float dx = x - closestX;
    float dy = y - closestY;
    float distance = std::sqrt(dx * dx + dy * dy);

    return distance < radius + width * 0.5f;
}
