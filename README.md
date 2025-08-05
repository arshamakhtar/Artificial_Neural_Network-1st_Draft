# Artificial Neural Network 1st Draft

This is a custom implementation of an Artificial Neural Network written in C++14. It is built from scratch without using any external libraries or machine learning frameworks.

## 🚀 Features

- Fully connected feedforward neural network
- Backpropagation with adjustable learning rate, momentum, and bias
- Customizable topology
- Lightweight and academic-oriented
- Built using only standard C++ (C++14)

## ⚙️ Structure

- `src/` – Implementation source files
- `include/` – Header files defining classes like `Neuron`, `Layer`, `Matrix`, and utilities
- `main.cpp` – Example usage and training loop
- `CMakeLists.txt` – Build configuration

## 🧠 Topology Example

The network topology can be defined like:
```cpp
vector<int> topology = {3, 2, 3};
```
Where:
- `3` input neurons
- `2` hidden neurons
- `3` output neurons

## 📈 Training

The training loop feeds forward the input, calculates errors, and applies backpropagation:
```cpp
for(int i = 0; i < 1000; i++) {
    nn->feedForward();
    nn->setErrors();
    nn->backPropogation();
}
```

## ⚠️ Known Limitations

- ❌ **No memory management**: The current version does not handle freeing dynamically allocated memory (e.g., `new` is used without `delete`). This may lead to memory leaks.
- 🧪 **Academic use only**: This is a first-draft implementation intended for learning and experimentation purposes, not for production use.
- ⛔ **No file I/O or persistence**: Weights are not saved or loaded.
- 🧱 **Not optimized**: There are no performance or concurrency optimizations.

## 📋 Requirements

- C++14 compatible compiler (e.g., `g++ -std=c++14`)
- CMake (optional, for building via `CMakeLists.txt`)

## 🧪 Sample Output

```
Epoch:999
Total error: 0.117724
====================
OUTPUT: 0.943096 0.00054734 0.943094
TARGET: 1 0 1
```

## 🧰 Future Improvements

- Add memory management (e.g., smart pointers)
- Save/load network state
- GUI or CLI for live training
- Error plotting

## 📚 License

This project is for educational purposes and is not licensed for commercial use.
