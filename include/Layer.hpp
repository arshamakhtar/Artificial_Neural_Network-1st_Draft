#ifndef _LAYER_HPP_
#define _LAYER_HPP_

#include<iostream>
#include<vector>
#include "Neuron.hpp"
#include "Matrix.hpp"
using namespace std;

class Layer{
    public:

        Layer(int size);
        void setVal(int i, double v);

        Matrix *matrixifyVals();
        Matrix *matrixifyActivatedVals();
        Matrix *matrixifyDerivedVals();

        vector<Neuron *> getNeurons(){
            return this->neurons;
        };
        void setNeurons(vector<Neuron *> neurons){
            this-> neurons = neurons;
        }
    private:
        //total neurons present in the layer    
        int size;
        vector<Neuron *> neurons;
       
};

#endif
