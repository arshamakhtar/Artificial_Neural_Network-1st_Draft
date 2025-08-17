# Dimensionality Reduction Neural Network (Autoencoder)

This is a custom implementation of a **Dimensionality Reduction Autoencoder** written in **C++14**.  
It is built entirely from scratch without external ML libraries or frameworks.  
The C++ implementation can leverage GPU parallelism for scalable, fast computation, making it suitable for experimentation with compression of high-dimensional data.

## 🚀 Features

- Fully connected **autoencoder** (feedforward + backpropagation)  
- **Dimensionality reduction via bottleneck layer**  
- Adjustable learning rate, momentum, and bias  
- Customizable topology for different compression ratios  
- Lightweight, academic-oriented, and written in pure **C++14**

## ⚙️ Structure

- `src/` – Source files for the autoencoder  
- `include/` – Header files (`Neuron`, `Layer`, `Matrix`, etc.)  
- `main.cpp` – Example: reducing 625-dimensional data to 30 latent features  
- `CMakeLists.txt` – Build configuration  

## 🧠 Topology Example

Define the network architecture for dimensionality reduction:


vector<int> topology = {625, 100, 30, 100, 625};

## 📉 Dimensionality Reduction Workflow

The autoencoder trains by learning to reconstruct its input.  
The bottleneck layer acts as the compressed representation:

```cpp
for (int epoch = 0; epoch < 1000; epoch++) {
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
Total error: 0.0117724
====================
OUTPUT: 0.953096 0.00054734 0.953094
TARGET: 1 0 1
```

## 🧰 Future Improvements

- Add memory management (e.g., smart pointers)
- Save/load network state
- GUI or CLI for live training
- Error plotting

## 📚 License

This project is for educational purposes and is not licensed for commercial use.
