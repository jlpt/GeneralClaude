#include "player.h"
#include "raylib.h"
#include <math.h>
Player::Player(int x, int y, int size, Camera2D& camera) : x{ x }, y{ y }, size{ size }, camera{ camera } {

}

void Player::Draw() {
	//DrawCircle(x, y, size, WHITE);

	//DrawCircle(x, y, size-2, RED);
	

}

void Player::Update() {
    float speed = 2.0f; // Base speed for movement
    float moveX = 0.0f, moveY = 0.0f;

    // Determine movement direction
    if (IsKeyDown(KEY_W)) moveY -= 1; // Move up
    if (IsKeyDown(KEY_S)) moveY += 1; // Move down
    if (IsKeyDown(KEY_A)) moveX -= 1; // Move left
    if (IsKeyDown(KEY_D)) moveX += 1; // Move right

  
    // Apply the movement
    x += moveX * speed;
    y += moveY * speed;
}

