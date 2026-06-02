# Adult Income Prediction: Linear Regression vs SVM vs Neural Network

## Abstract
This project applies theoretical concepts from my Machine Learning course by implementing three models on the UCI Adult Income dataset: Linear Regression, Support Vector Machine (SVM), and a shallow Neural Network. The task is binary classification — predicting whether an individual's annual income exceeds $50K. All three models achieved similar test accuracy (84–85%), with linear regression falling just 1% short of the neural network. This result suggests that the underlying decision boundary for this dataset is approximately linear.

## Motivation
This project demonstrates:
- Implementation of three fundamental ML models from scratch (using scikit-learn and PyTorch)
- Systematic comparison on a real-world tabular dataset
- Handling of categorical features, missing data, and class imbalance
- Understanding of when simple models are sufficient

## Dataset
- **Source**: UCI Adult Income (also known as "Census Income")
- **Samples**: 48,842
- **Features**: 14 (age, education, occupation, hours-per-week, etc.)
- **Target**: Binary (≤50K / >50K)
- **Class distribution**: ~76% ≤50K, ~24% >50K (imbalanced)

## Models & Implementation

| Model | Library | Key Details |
|-------|---------|--------------|
| Linear Regression | NumPy, pandas | Implemented from scratch using no ML library |
| SVM | scikit-learn | RBF kernel, C=1 |
| Neural Network | PyTorch | 1 hidden layers, ReLU, binary cross-entropy loss |

## Results

| Model | Test Accuracy |
|-------|---------------|
| Linear Regression | **84%** |
| SVM | **85%** |
| Neural Network | **85%** |

**Key observation**: The neural network and SVM do not significantly outperform linear regression. This indicates that the relationship between features and income is approximately linear. It's not sufficient to use a neural network or SVM here while we can solve the problem using linear regression which leads us towards less computing complexity.

## Discussion

All three models achieved 84–85% accuracy on the Adult Income dataset. The neural network — with one hidden layer and SVM — did not outperform linear regression. 

**What I learned from this:**
- Adding complexity (more layers, advanced optimizers) doesn't guarantee better results.
- A simple model trained in 0.2 seconds can be as useful as one taking 45 seconds.
- The model's performance depends on the data too , just improving its power is not gonna make it better

**For my future research:** I want to understand what makes some datasets require non-linear models while others don't. This project raised that question for me.

## Setup & Reproduction

### Requirements
- Python 3.9+
- numpy, pandas, scikit-learn, torch, matplotlib
- See `requirements.txt`

