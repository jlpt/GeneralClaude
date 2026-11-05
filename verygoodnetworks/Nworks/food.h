#include "entity.h"
#pragma once
class Food : public Entity{
public:
	int x;
	int y;
	int size;
	virtual void Draw();
	Food(int x, int y, int size);
	
};