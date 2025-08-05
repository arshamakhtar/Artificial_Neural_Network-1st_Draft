#ifndef _NEURAL_NETWORK_HPP_
#define _NEURAL_NETWORK_HPP_

#include<iostream>
#include<vector>
#include <algorithm>
#include "utils/MultipyMatrix.hpp"
#include "Matrix.hpp"
#include "Layer.hpp"

using namespace std;

class NeuralNetwork{
    public:

        NeuralNetwork(vector<int> topology);
        void setCurrentInput(vector<double> input);
        void setCurrentTarget(vector<double> target) { this-> target = target; };
        void printToConsole();
        void feedForward();
        void setErrors();
        void backPropogation();

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

        double getTotalError(){
            return this-> error;
        };
        vector<double> getErrors(){
            return this->errors;
        };

    private:
        //order of Layers    
        int                 topologySize;
        vector<int>         topology;
        vector<Layer *>     Layers;
        vector<Matrix *>    weightMatrices;
        vector<Matrix *>    gradientMatrices;
        vector<double>      input;
        vector<double>      errors;
        double              error;
        vector<double>      target;
        vector<double>      historicalErrors;
};

#endif
