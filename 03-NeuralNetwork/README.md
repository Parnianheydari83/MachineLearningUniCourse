# Adult Income Prediction – Neural Network (PyTorch)

## 🔎 Abstract

This project applies a feedforward neural network to the UCI Adult Income dataset to predict whether an individual earns more than $50K per year. The dataset was preprocessed by handling missing values, one-hot encoding categorical features, standardizing numerical features, and performing an 80/20 train-test split. The neural network was implemented using PyTorch with one hidden layer containing 512 neurons and trained using Stochastic Gradient Descent (SGD) with Cross-Entropy Loss. Model performance was monitored throughout training by tracking both training loss and train/test accuracy over 100 epochs.

## ✨ Motivation

* Learn how neural networks perform binary classification on structured tabular data.
* Understand the complete deep learning workflow using PyTorch.
* Compare neural network performance with previous Logistic Regression and Support Vector Machine (SVM) models.
* Gain hands-on experience with forward propagation, backpropagation, gradient descent, and mini-batch training.

## 🧠 Model and Implementation

| Component                  | Implementation                                                                                   | Libraries         |
| -------------------------- | ------------------------------------------------------------------------------------------------ | ----------------- |
| **Data loading**           | Loaded the Adult Income dataset directly from the UCI repository                                 | `pandas`          |
| **Missing value handling** | Replaced `?` with `NaN` and removed incomplete rows                                              | `pandas`, `numpy` |
| **Target encoding**        | Mapped `<=50K` → 0 and `>50K` → 1                                                                | `pandas`          |
| **Feature encoding**       | One-hot encoding using `pd.get_dummies()`                                                        | `pandas`          |
| **Train/test split**       | Custom randomized 80/20 split using shuffled indices                                             | `numpy`           |
| **Feature scaling**        | Standardized features using the training set mean and standard deviation                         | `numpy`           |
| **Tensor conversion**      | Converted NumPy arrays into PyTorch tensors                                                      | `PyTorch`         |
| **Data loading**           | Created mini-batches with `TensorDataset` and `DataLoader`                                       | `PyTorch`         |
| **Neural network**         | Feedforward network with one hidden layer (512 neurons), ReLU activation, and two output neurons | `PyTorch`         |
| **Loss function**          | Cross-Entropy Loss                                                                               | `PyTorch`         |
| **Optimizer**              | Stochastic Gradient Descent (SGD) with learning rate of 0.01                                     | `PyTorch`         |
| **Training**               | Trained for 100 epochs using mini-batch gradient descent and backpropagation                     | `PyTorch`         |
| **Evaluation**             | Measured training and test accuracy after each epoch                                             | `PyTorch`         |
| **Visualization**          | Training loss curve and train/test accuracy curves                                               | `matplotlib`      |

### Requirements

* Python 3.9+
* See `requirements.txt`
