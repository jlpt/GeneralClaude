#pragma once
#include "module.h"
#include "organism.h"
class NeuralNetwork;
class TeleportModule : public Module {
public:
	NeuralNetwork* network;
	Organism* organism;
	TeleportModule();
	TeleportModule(NeuralNetwork* network = nullptr, Organism* organism = nullptr);
	virtual std::unique_ptr<Module> Clone(Organism* newOrganism, NeuralNetwork* newNetwork) const override;
	virtual void Update() override;


};



