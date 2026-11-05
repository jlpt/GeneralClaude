

#pragma once
#include "entity.h"
#include "raylib.h"
class Player : public Entity {
public:
	int x;
	int y;
	int size;
	Camera2D& camera;
	Player(int x, int y, int size, Camera2D& camera);
	void Draw();
	void Update();

};