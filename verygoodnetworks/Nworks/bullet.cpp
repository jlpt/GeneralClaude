#include "bullet.h"
#include "raylib.h";

Bullet::Bullet(int x, int y, int size, int xvel, int yvel) : x{ x }, y{ y }, size{ size }, xvel{ xvel }, yvel{ yvel } {

}

void Bullet::Draw() {
	DrawCircle(x, y, size, LIME);
	//DrawCircle(x, y, 45, Color{ 0,82,172,64 });
}

void Bullet::Update() {
	x += xvel;
	y += yvel;
}