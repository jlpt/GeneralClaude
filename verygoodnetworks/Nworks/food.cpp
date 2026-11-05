#include "food.h"
#include "raylib.h";

Food::Food(int x, int y, int size) : x{ x }, y{ y }, size{ size } {

}

void Food::Draw() {
	DrawCircle(x, y, size + 3, BLACK);
	DrawCircle(x, y, size, RED);

}