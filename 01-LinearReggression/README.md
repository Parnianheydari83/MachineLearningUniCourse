# Adult Income Prediction – Linear Regression (from scratch)

## 🔎Abstract
This project applies linear regression to the UCI Adult Income dataset to predict whether an individual earns more than $50K.Unlike typical implementations that rely on scikit-learn, I implemented the model from scratch using NumPy and pandas .After preprocessing (dropping missing values, one‑hot encoding, stratified 80/20 split), the model achieved 84% test accuracy. The result served as a baseline to compare the model against more complex models(SVM and Neural Network) Which surprisingly they only reached 85% .

## ✨Motivation
- Bridging theory and practice - moving from whiteboard equations to working, evaluatable code that runs on actual census data.
- Working with real data which is messy 
- Understanding the fundamental behind the code instead of using pre-built libraries


## 🧠Model and Implementation 
| Component | What I implemented | Libraries used (only for data handling & visualization) |
|-----------|--------------------|----------------------------------------------------------|
| **Data loading** | Read CSV from local file | `pandas` |
| **Missing value handling** | Replaced `?` with `'missing'` (kept rows, encoded as category) | `pandas`, `numpy` |
| **Column removal** | Dropped `native.country` due to high cardinality | `pandas` |
| **Target encoding** | Mapped `<=50K` → 0, `>50K` → 1, converted to boolean | `pandas` |
| **Feature encoding** | One‑hot encoding via `pd.get_dummies()` | `pandas` |
| **Feature scaling** | Standardization: (x - mean) / std on numeric columns only | `numpy` |
| **Boolean to int** | Converted boolean columns to 0/1 integers | `pandas` |
| **Train/test split** | Implemented custom function: shuffled indices, 80/20 split, fixed seed | **Only `numpy`** |
| **Sigmoid function** | σ(z) = 1 / (1 + e⁻ᶻ) | `numpy` |
| **Cost function** | Binary cross‑entropy (log loss) from scratch | `numpy` |
| **Gradient descent** | One‑step update: dw = (1/m)·Xᵀ·(ŷ−y), db = (1/m)·Σ(ŷ−y) | `numpy` |
| **Training loop** | Iterative parameter updates for 1000 epochs, tracked cost & accuracy | `numpy` |
| **Prediction** | ŷ = sigmoid(X·θ + b), threshold at 0.5 | `numpy` |
| **Accuracy metric** | Custom `accuracy_score` (comparison + mean) | `numpy` |
| **Visualization** | Loss curve + accuracy over epochs | `matplotlib` |
### Requirements
- Python 3.9+
- See `requirements.txt`

