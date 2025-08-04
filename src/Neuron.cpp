#include "../include/Neuron.hpp"

void Neuron::setVal(double val){
    this -> val = val;
    activate();
    derive();
}

//Constructor
Neuron::Neuron(double val) {
    this->val = val;
    activate();
    derive();
}

//Fast Sigmoid function
void Neuron::activate() {
    this->activatedVal = this->val / (1 + std::abs(this->val));
}

//Derivative for Fast Sigmoid Function
void Neuron::derive() {
    this->derivedVal = this->activatedVal * (1 - this->activatedVal);
}

