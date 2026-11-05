#ifndef BLOCK_H
#define BLOCK_H
#include "entity.h"
#include "raylib.h"
class Block : public Entity {
protected:
	int x;
	int y;
	int size;
	Color color;

public:
	Block(int x, int y, int size, Color color);
	Block();
	void Draw();

};

#endif