#include "..\include\NeuralNetwork.hpp"

void NeuralNetwork::backPropogation(){
    vector<Matrix *> newWeights;
    Matrix* gradient;

    // output to hidden 
    int outputLayerIndex         = this -> Layers.size() -1;
    Matrix *derivedValuesYtoZ    = this-> Layers.at(outputLayerIndex)->matrixifyDerivedVals();
    Matrix *gradientsYtoZ        = new Matrix(1, this -> Layers.at(outputLayerIndex)-> getNeurons().size(), false);
    for(int i = 0; i < this-> errors.size() ;  ++i){
        double d = derivedValuesYtoZ -> getValue(0, i);
        double e = this-> errors.at(i);
        double g = d * e;
        gradientsYtoZ -> setValue(0, i, g);
    }
    int lastHiddenLayerIndex        = outputLayerIndex - 1;
    Matrix *weightsOutputToHidden   = this-> weightMatrices.at(lastHiddenLayerIndex);
    Layer *lastHiddenLayer          = this->Layers.at(lastHiddenLayerIndex);
    Matrix *deltaOtuputToHidden     = ( new utils::MultiplyMatrix(
                                        gradientsYtoZ-> transpose(), 
                                        lastHiddenLayer->matrixifyActivatedVals()))
                                        ->execute()->transpose();

    Matrix *newWeightsOutputToHidden = new Matrix(
                                                    deltaOtuputToHidden-> getNumRows(), 
                                                    deltaOtuputToHidden-> getNumCols(), 
                                                    false
                                                );
    for(int r =0; r < deltaOtuputToHidden -> getNumRows() ; ++r){
        for(int c= 0; c < deltaOtuputToHidden -> getNumCols() ; c++){
            double originalWeight   = weightsOutputToHidden-> getValue(r, c);
            double deltaWeight      = deltaOtuputToHidden->   getValue(r, c);
            newWeightsOutputToHidden-> setValue(r, c, (originalWeight-deltaWeight));
        }
    }

    newWeights.push_back(newWeightsOutputToHidden);
    gradient = new Matrix( gradientsYtoZ->getNumRows() ,  gradientsYtoZ->getNumCols() , false);
    for(int r =0; r< gradientsYtoZ -> getNumRows() ; ++r){
        for(int c= 0; c < gradientsYtoZ -> getNumCols() ; ++c){
            gradient-> setValue(r , c, gradientsYtoZ->getValue(r, c));
        }
    }

    // moving from last hidden layer to input layer
    for( int i = (outputLayerIndex -1); i > 0; i--) {
        Layer *l                = this-> Layers.at(i);
        Matrix *derivedHidden   = l->matrixifyDerivedVals();
        Matrix *activatedHidden = l->matrixifyActivatedVals();
        Matrix *derivedGradients= new Matrix(
                                             1,
                                             l->getNeurons().size(),
                                             false
                                            );
        Matrix *weightMatrix        = this-> weightMatrices.at(i);
        Matrix *originalWeight      = this-> weightMatrices.at(i - 1);
        
        for(int r = 0 ; r < weightMatrix -> getNumRows(); ++r){
            double sum = 0.0;
            for(int c = 0; c< weightMatrix -> getNumCols(); ++c){
                double p = gradient->getValue(0, c)* weightMatrix -> getValue(r , c);
                sum += p;
            }
            double g = sum * activatedHidden-> getValue(0, r);
            derivedGradients -> setValue( 0, r, g);
        }

        Matrix *leftNeurons      = (i - 1)== 0 ? this->Layers.at(0)-> matrixifyVals(): this-> Layers.at(i - 1)->matrixifyActivatedVals();
        Matrix *deltaWeights     = (new utils:: MultiplyMatrix(derivedGradients-> transpose(), leftNeurons))->execute()->transpose();
        Matrix *newWeightsHidden = new Matrix (deltaWeights-> getNumRows(), deltaWeights-> getNumCols() , false);

        for(int r = 0; r < newWeightsHidden-> getNumRows() ; ++r){
            for(int c = 0; c< newWeightsHidden-> getNumCols(); ++c){
                double w = originalWeight-> getValue(r, c);
                double d = deltaWeights -> getValue(r, c);
                double n = w - d;
                newWeightsHidden->setValue(r , c , n);
            }
        }
        
        gradient = new Matrix(derivedGradients-> getNumRows(),derivedGradients-> getNumCols(), false);
        for(int r = 0; r < derivedGradients->getNumRows(); ++r){
            for(int c = 0; c < derivedGradients->getNumCols(); ++c){
                gradient-> setValue(r, c, derivedGradients->getValue(r, c));
            }
        }
        newWeights.push_back(newWeightsHidden);
    }
    //cout<< " done with back prop "<<endl;
    //cout<< "New weights Size: " <<newWeights.size() << endl;
    //cout<< "Old weights Size: " << this-> weightMatrices.size() << endl;
    
    reverse(newWeights.begin(), newWeights.end());
    this->weightMatrices =newWeights;
}

void NeuralNetwork::printInputToConsole(){
    for(int i = 0; i < this->input.size() ; i++){
        cout<<this-> input.at(i) << "\t";
    }
    cout<<endl;
}
void NeuralNetwork::printTargetToConsole(){
    for(int i = 0; i < this->target.size() ; i++){
        cout<<this-> target.at(i) << "\t";
    }
    cout<<endl;
}
void NeuralNetwork::printHistoricalErrors(){
    for(int i=0; i < this->historicalErrors.size(); ++i){
        cout<< this-> historicalErrors.at(i);
        if(i != this->historicalErrors.size() - 1){
            cout<< ", ";
        }
    }
    cout<<endl;
}
void NeuralNetwork::printOutputToConsole(){
    int indexOfOutputLayer  = this->Layers.size() - 1;
    Matrix *outputValues    = this->Layers.at(indexOfOutputLayer)->matrixifyActivatedVals();
    for(int c = 0; c < outputValues -> getNumCols() ; ++c){
        cout << outputValues -> getValue(0, c)<< "\t";
    }
    cout<< endl;
}

void NeuralNetwork::setErrors(){
    errors.clear();
    if(this->target.size() == 0){
        cerr << "No Target for this Neural Network" << endl;
        assert(false);
    }

    if(this-> target.size() != this-> Layers.at(this-> Layers.size() - 1)-> getNeurons().size()){
        cerr<< "Target size is not the same as the output Layer size: " << this->Layers.at(this-> Layers.size()- 1)->getNeurons().size() << endl;
        assert(false);
    }

    this -> error = 0.00;
    int outputLayerIndex = (this-> Layers.size() - 1);
    vector<Neuron *> outputNeurons = this-> Layers.at(outputLayerIndex) -> getNeurons();
    for(int i = 0; i < target.size() ; ++i){
        double tempErr = (outputNeurons.at(i) -> getActivatedVal() - target.at(i));
        errors.push_back(tempErr);
        this -> error += pow(tempErr, 2);
    }
    this->error = 0.5 * this->error;

    historicalErrors.push_back(this -> error);
}

void NeuralNetwork::feedForward(){
    for(int i = 0; i< (this->Layers.size() - 1); ++i){
        Matrix *a = this->getNeuronMatrix(i);

        if(i != 0){
            a = this->getActivatedNeuronMatrix(i);
        }

        Matrix *b = this->getWeightMatrix(i);
        Matrix *c = (new utils::MultiplyMatrix(a, b))-> execute();

        for(int c_index = 0; c_index < c-> getNumCols(); c_index++){
            this->setNeuronValue(i + 1, c_index , c->getValue(0, c_index));
        }
    }
}

void NeuralNetwork::setCurrentInput(vector<double> input){
    this-> input = input;
    for(int i = 0; i < input.size(); ++i ){
    this-> Layers.at(0)->setVal( i, input.at(i) );
    }
}

NeuralNetwork::NeuralNetwork(vector<int> topology){
    this-> topology     = topology;
    this-> topologySize = topology.size();
    for(int i =0; i < topologySize ; ++i){
        Layer *l=  new Layer( topology.at(i) );
        this -> Layers.push_back(l);
    }

    for(int i = 0; i < (topologySize-1) ; ++i){
        Matrix *m = new Matrix( topology.at(i), topology.at( i + 1 ), true);
        this->weightMatrices.push_back(m);
    }
}

void NeuralNetwork::printToConsole(){
    for(int i = 0; i < this->Layers.size(); ++i){
        cout<< "LAYER:" << i <<endl;
        if( i == 0 ){
            Matrix *m = this->Layers.at(i) -> matrixifyVals();
            m->printToConsole();
        }else {
            Matrix *m= this-> Layers.at(i) -> matrixifyActivatedVals();
            m -> printToConsole();
        }
        cout<<"---------------------"<<endl;
        if(i < this-> Layers.size() - 1){
            cout<< "Weight Matrix: "<< i <<endl;
            this-> getWeightMatrix(i) -> printToConsole();
        }
        cout<<"---------------------"<<endl;
        
    }
}
