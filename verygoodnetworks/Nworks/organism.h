#pragma once
#include "neural_network.h"
#include "is_clickable.h"
#include "food_sensor.h"
#include <vector>
#include "raylib.h"
#include <memory>
class Organism : public IsClickable ,public Entity{

	
public: 
	float x;
	float y;
	int size;
	int input;
	int output;
	float health;
	float air;
	float mutationRate = 1.0f;
	bool alive = true;
	int foodEaten = 0;
	int generation = 0;
	int inputsUsed = 0;
	int outputsUsed = 0;
	Color color;
	std::unique_ptr<NeuralNetwork> network;
	std::vector<std::unique_ptr<Module>> modules;
	Organism(float x, float y, int size);


	
	void InitNetwork(int neurons, int connections, int inputs, int outputs);
	void AddModule(std::unique_ptr<Module> module);

	void Draw();
	void Update();
	void DrawNetwork();
	
	void Click() override;
	std::unique_ptr<Organism> Reproduce();

};