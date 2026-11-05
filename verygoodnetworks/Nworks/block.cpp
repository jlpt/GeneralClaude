#include "block.h"
#include "raylib.h"
Block::Block(int x, int y, int size, Color color) : x{ x }, y{ y }, size{ size }, color{ color } {}
Block::Block(){}
void Block::Draw() {
	DrawRectangle(x, y, size, size, color);
}