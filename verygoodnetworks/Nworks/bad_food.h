
#pragma once
#include "entity.h"
class BadFood : public Entity {
public:
	int x;
	int y;
	int size;
	void Draw();
	BadFood(int x, int y, int size);

};