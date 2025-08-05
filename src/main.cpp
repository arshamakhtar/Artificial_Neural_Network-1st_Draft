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

    vector<int> topology={3,2,3};
    
    NeuralNetwork *nn= new NeuralNetwork(topology);
    nn->setCurrentInput(input);
    nn->setCurrentTarget(input);

    //training process
    for(int i=0; i<30; i++){
        cout << "Epoc:" <<i<<endl;
        nn->feedForward();
        nn->setErrors();
        cout<< "Total error: "<< nn->getTotalError() << endl;
        nn->backPropogation();
    }
    return 0;
}
