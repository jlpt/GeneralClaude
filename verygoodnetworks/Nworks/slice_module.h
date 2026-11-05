#pragma once
#include "module.h"
#include "organism.h"
#include "slice.h"
class NeuralNetwork;
class SliceModule : public Module {
public:
	NeuralNetwork* network;
	Organism* organism;
	std::vector<Slice>* slices;
	int cooldownPeriod = 10;
	int cooldown = cooldownPeriod;
	SliceModule();
	SliceModule(NeuralNetwork* network = nullptr, Organism* organism = nullptr, std::vector<Slice>* slices = nullptr);
	virtual std::unique_ptr<Module> Clone(Organism* newOrganism, NeuralNetwork* newNetwork) const override;
	virtual void Update() override;
	
};



