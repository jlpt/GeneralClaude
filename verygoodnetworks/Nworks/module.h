#pragma once
#include <memory>

class Organism;
class World;

// A body part that connects the brain to the world. Sensors write to input
// neurons, actuators read output neurons. Organism::AddModule gives each
// module its own block of inputs/outputs, starting at firstInput/firstOutput.
class Module {
public:
    virtual ~Module() = default;
    virtual std::unique_ptr<Module> Clone() const = 0;

    virtual int InputCount() const { return 0; }
    virtual int OutputCount() const { return 0; }
    virtual void Sense(Organism& /*self*/, const World& /*world*/) {}  // write inputs
    virtual void Act(Organism& /*self*/, World& /*world*/) {}          // read outputs

    int firstInput = 0;
    int firstOutput = 0;
};

// Base class that implements Clone() by copying the concrete module.
template <typename Derived>
class ModuleBase : public Module {
public:
    std::unique_ptr<Module> Clone() const override {
        return std::make_unique<Derived>(static_cast<const Derived&>(*this));
    }
};
