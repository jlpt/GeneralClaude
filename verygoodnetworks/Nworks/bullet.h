#pragma once
#include "entity.h"
class Bullet : public Entity {
public:
	int x;
	int y;
	int size;
	int xvel;
	int yvel;
	void Draw();
	void Update();
	Bullet(int x, int y, int size, int xvel, int yvel);

};