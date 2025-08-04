#ifndef _NEURAL_NETWORK_HPP_
#define _NEURAL_NETWORK_HPP_

#include<iostream>
#include<vector>
#include "utils/MultipyMatrix.hpp"
#include "Matrix.hpp"
#include "Layer.hpp"

using namespace std;

class NeuralNetwork{
    public:

        NeuralNetwork(vector<int> topology);
        void setCurrentInput(vector<double> input);
        void printToConsole();
        void feedForward();

        Matrix* getNeuronMatrix( int index ){
            return this->Layers.at(index)-> matrixifyVals();
        };

        Matrix* getActivatedNeuronMatrix( int index ){
            return this->Layers.at(index)-> matrixifyActivatedVals();
        };

        Matrix* getDerivedNeuronMatrix( int index ){
            return this->Layers.at(index)-> matrixifyDerivedVals();
        };

        Matrix *getWeightMatrix(int index) {
            return new Matrix(*this->weightMatrices.at(index));
        };

        void setNeuronValue(int indexLayer, int indexNeuron, double val){
            this->Layers.at(indexLayer)->setVal(indexNeuron, val);
        };

    private:
        //order of Layers    
        int                 topologySize;
        vector<int>         topology;
        vector<Layer *>     Layers;
        vector<Matrix *>    weightMatrices;
        vector<double>      input;
};

#endif
