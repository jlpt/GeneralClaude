#include <iostream>
#include <cmath>
#include <algorithm>
#include "neural_network.h"
#include "slice_sensor.h"
#include "slice.h"
#include "organism.h"

// Helper function: returns the distance from a point (px, py)
// to the line segment defined by (x1, y1) and (x2, y2)
static float PointToLineSegmentDistance(float px, float py, float x1, float y1, float x2, float y2) {
    float lx = x2 - x1;
    float ly = y2 - y1;
    float lenSq = lx * lx + ly * ly;
    if (lenSq == 0.0f) return std::sqrt((px - x1) * (px - x1) + (py - y1) * (py - y1));
    float t = ((px - x1) * lx + (py - y1) * ly) / lenSq;
    t = std::max(0.0f, std::min(1.0f, t));
    float projX = x1 + t * lx;
    float projY = y1 + t * ly;
    float dx = px - projX;
    float dy = py - projY;
    return std::sqrt(dx * dx + dy * dy);
}

// Constructor
SliceSensor::SliceSensor(NeuralNetwork* network, Organism* organism, std::vector<Slice>* sliceList)
    : network{ network }, organism{ organism }, sliceList{ sliceList } {
    input = 8;  // or adjust as needed
    output = 0;
}

SliceSensor::SliceSensor() {
    input = 8;
    output = 0;
}

std::unique_ptr<Module> SliceSensor::Clone(Organism* newOrganism, NeuralNetwork* newNetwork) const {
    return std::make_unique<SliceSensor>(newNetwork, newOrganism, sliceList);
}

// Update method: detects the closest slice and updates directional neuron activations
// for each endpoint separately.
void SliceSensor::Update() {
    if (network == nullptr || organism == nullptr || sliceList == nullptr) {
        std::cerr << "SliceSensor not properly initialized!" << std::endl;
        return;
    }

    const float sensorRange = 600.0f;
    Slice* closestSlice = nullptr;
    float closestDist = sensorRange;

    // Find the closest slice by measuring distance from the organism
    // to each slice's line segment.
    for (Slice& slice : *sliceList) {
        float distance = PointToLineSegmentDistance(organism->x, organism->y, slice.x1, slice.y1, slice.x2, slice.y2);
        if (distance < closestDist) {
            closestDist = distance;
            closestSlice = &slice;
        }
    }

    // Initialize activations for both endpoints.
    float aLeft1 = 0.0f, aRight1 = 0.0f, aUp1 = 0.0f, aDown1 = 0.0f;
    float aLeft2 = 0.0f, aRight2 = 0.0f, aUp2 = 0.0f, aDown2 = 0.0f;

    if (closestSlice != nullptr) {
        // For endpoint 1
        float deltaX1 = closestSlice->x1 - organism->x;
        float deltaY1 = closestSlice->y1 - organism->y;
        if (deltaX1 < 0) {
            aLeft1 = (sensorRange - std::fabs(deltaX1)) / sensorRange;
            aLeft1 = std::min(aLeft1, 1.0f);
        }
        else {
            aRight1 = (sensorRange - std::fabs(deltaX1)) / sensorRange;
            aRight1 = std::min(aRight1, 1.0f);
        }
        if (deltaY1 < 0) {
            aUp1 = (sensorRange - std::fabs(deltaY1)) / sensorRange;
            aUp1 = std::min(aUp1, 1.0f);
        }
        else {
            aDown1 = (sensorRange - std::fabs(deltaY1)) / sensorRange;
            aDown1 = std::min(aDown1, 1.0f);
        }

        // For endpoint 2
        float deltaX2 = closestSlice->x2 - organism->x;
        float deltaY2 = closestSlice->y2 - organism->y;
        if (deltaX2 < 0) {
            aLeft2 = (sensorRange - std::fabs(deltaX2)) / sensorRange;
            aLeft2 = std::min(aLeft2, 1.0f);
        }
        else {
            aRight2 = (sensorRange - std::fabs(deltaX2)) / sensorRange;
            aRight2 = std::min(aRight2, 1.0f);
        }
        if (deltaY2 < 0) {
            aUp2 = (sensorRange - std::fabs(deltaY2)) / sensorRange;
            aUp2 = std::min(aUp2, 1.0f);
        }
        else {
            aDown2 = (sensorRange - std::fabs(deltaY2)) / sensorRange;
            aDown2 = std::min(aDown2, 1.0f);
        }
    }
    // If no slice is found, leave activations as zero (or set to a baseline if desired)

    // Update the network's neurons.
    // For endpoint 1: neurons 0 (left), 1 (right), 2 (up), 3 (down)
    network->neurons[0 + inputStartingNeuron].activation = aLeft1;
    network->neurons[1 + inputStartingNeuron].activation = aRight1;
    network->neurons[2 + inputStartingNeuron].activation = aUp1;
    network->neurons[3 + inputStartingNeuron].activation = aDown1;
    // For endpoint 2: neurons 4 (left), 5 (right), 6 (up), 7 (down)
    network->neurons[4 + inputStartingNeuron].activation = aLeft2;
    network->neurons[5 + inputStartingNeuron].activation = aRight2;
    network->neurons[6 + inputStartingNeuron].activation = aUp2;
    network->neurons[7 + inputStartingNeuron].activation = aDown2;
}
