#ifndef MODULE_H
#define MODULE_H

#include <memory>

class Organism;      // Forward declaration
class NeuralNetwork; // Forward declaration

class Module {
public:
    int input;
    int output;
    int inputStartingNeuron;
    int outputStartingNeuron;
    virtual ~Module() = default;

    // Clone creates a new copy of the module with updated pointers.
    virtual std::unique_ptr<Module> Clone(Organism* newOrganism, NeuralNetwork* newNetwork) const = 0;

    virtual void Update() = 0;
   
      
};

#endif // MODULE_H
