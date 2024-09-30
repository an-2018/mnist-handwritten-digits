# Handwritten Digit Recognition with Neural Networks (MNIST OCR)

## Project Overview

This project implements a neural network to recognize handwritten digits from the MNIST dataset. The aim is to create a proof-of-concept model for Optical Character Recognition (OCR) systems using deep learning. This project demonstrates the end-to-end process of dataset preprocessing, building, training, and tuning a neural network to achieve a classification accuracy of at least 90% on the test set.

## Project Files

- `handwritten_digit_recognition.ipynb`: The Jupyter Notebook containing the code for preprocessing, model building, training, evaluation, and saving the model.
- `trained_model.pth`: The saved trained model, which can be loaded for future inference or analysis.

## Project Instructions

The project is divided into several steps:

### Step 1: Dataset Loading and Preprocessing

- **Dataset**: The MNIST handwritten digit dataset is loaded from `torchvision.datasets`.
- **Preprocessing**: The dataset is converted to PyTorch tensors and normalized. Input images are flattened into vectors for input to the neural network.
- **DataLoader**: A PyTorch DataLoader is used to batch the data for efficient training and testing.

### Step 2: Data Visualization and Exploration

- Visualize the dataset using a helper function. Explore the size and shape of the data both before and after preprocessing.
- Briefly explain and justify preprocessing choices, such as normalization and flattening.

### Step 3: Neural Network Model

- **Model**: A fully connected neural network is built using PyTorch to classify input digits.
- **Optimizer**: The model's weights are updated during training using a selected optimizer, such as stochastic gradient descent (SGD) or Adam.
- **Training**: The network is trained on the training set using the prepared DataLoader.

### Step 4: Evaluation and Model Tuning

- **Evaluation**: The accuracy of the model is evaluated on the test set.
- **Tuning**: Hyperparameters such as learning rate, batch size, and network architecture are tuned to achieve an accuracy of at least 90% on the test set.

### Step 5: Saving the Model

- The trained model is saved using `torch.save` for reuse in future applications.

## Requirements

This project requires the following dependencies:

- Python 3.x
- Jupyter Notebook
- PyTorch
- Torchvision
- Matplotlib (for visualization)

You can install the dependencies using `pip`:

```bash
pip install torch torchvision matplotlib
```

## How to Run

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-folder>
   ```

2. Open the Jupyter Notebook:
```bash
jupyter notebook handwritten_digit_recognition.ipynb
   ```

3. Run the notebook cells step-by-step to load the data, train the model, and evaluate its performance.
