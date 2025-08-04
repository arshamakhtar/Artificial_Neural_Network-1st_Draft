cmake_minimum_required(VERSION 3.7)
project(neural_network)

set(CMAKE_BUILD_TYPE        Debug)
set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -std=c++14")

#main app
add_executable(
    neural_network
    src/main.cpp
    src/Neuron.cpp
    src/Matrix.cpp
    src/Layer.cpp
    src/NeuralNetwork.cpp
    src/utils/MultiplyMatrix.cpp
    src/utils/MatrixToVector.cpp
)