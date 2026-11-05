#pragma once
#include <cmath>
#include "entity.h"
class Slice : public Entity {
public:
    float x1, y1, x2, y2;
    int width = 5;

    int frame = 0;
    int fadeInDuration = 10;
    int activeDuration = 1;
    int fadeOutDuration = 10;

    bool isDead = false;

    Slice(float x1, float y1, float x2, float y2);
    void Draw();
    void Update();
    bool IsActive() const;
    bool CollidesWith(float x, float y, float radius) const;

private:
    int GetOpacity() const;
};

/*
 for (auto it = slices.begin(); it != slices.end(); ) {
            it->Update();
            if (it->isDead) {
                it = slices.erase(it);
            }

            else {
                it->Draw();
                ++it;
            }
        }*/