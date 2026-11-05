#include "world_gen.h"
#include "block.h";
#include <vector>;

	
WorldGen::WorldGen(int size) {
	std::vector<Block> blocks(size);
	int x{ 0 };
	int y{ 16 };
	for (int i{ 0 }; i < size; i++) {
		Block block = Block(16 * i, 32, 16, WHITE);
		blocks.push_back(block);
	}
}
