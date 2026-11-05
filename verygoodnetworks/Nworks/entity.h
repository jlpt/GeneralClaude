#pragma once
class Entity {
public:
    virtual ~Entity() {}
    virtual void Update() {}
    virtual void Draw() {}
};
