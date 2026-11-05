#pragma once
#include "module.h"
#include "organism.h"
class NeuralNetwork;
class MovementModule : public Module {
public:
	NeuralNetwork* network;
	Organism* organism;
	MovementModule();
	MovementModule(NeuralNetwork* network = nullptr, Organism* organism = nullptr);
	virtual std::unique_ptr<Module> Clone(Organism* newOrganism, NeuralNetwork* newNetwork) const override;
	virtual void Update() override;
	
	
};



