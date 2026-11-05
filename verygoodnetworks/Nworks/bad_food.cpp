#include "bad_food.h"
#include "raylib.h";

BadFood::BadFood(int x, int y, int size) : x{ x }, y{ y }, size{ size } {

}

void BadFood::Draw() {
	DrawCircle(x, y, size + 3, BLACK);
	DrawCircle(x, y, size, BLUE);
	//DrawCircle(x, y, 500, Color{ 0,82,172,64 });
}