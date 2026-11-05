#ifndef MOVER_H
#define MOVER_H
#include "neural_network.h"

class Mover {
	Mover(int input, int output, NeuralNetwork* network);

	void Update();
};

#endif