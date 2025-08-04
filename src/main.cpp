#include<iostream>
#include "../include/Neuron.hpp"
#include "../include/Matrix.hpp"
#include "../include/NeuralNetwork.hpp"
#include "../include/utils/MultipyMatrix.hpp"

using namespace std;

int main(int argc, char **argv){
    vector<double> input;
    input.push_back(1.0);
    input.push_back(0.0);
    input.push_back(1.0);

    vector<int> topology={3,2,1};
    
    NeuralNetwork *nn= new NeuralNetwork(topology);
    nn->setCurrentInput(input);
    nn->feedForward();
    nn->printToConsole();

    return 0;
}
