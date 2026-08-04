# Adult Income Prediction – Support Vector Machine (SVM)

## 🔎 Abstract

This project applies a Support Vector Machine (SVM) classifier to the UCI Adult Income dataset to predict whether an individual earns more than $50K per year. The dataset was preprocessed by replacing missing values with `NaN`, removing incomplete rows, applying one-hot encoding to categorical features, and standardizing the feature values. An RBF-kernel SVM was trained using an 80/20 stratified train-test split and achieved approximately **85% test accuracy**. Model performance was evaluated using accuracy, a confusion matrix, and a classification report.

## ✨ Motivation

* Apply a powerful non-linear classification algorithm to a real-world dataset.
* Compare the performance of SVM with simpler models such as Logistic Regression.
* Learn how feature scaling and kernel methods improve classification performance.
* Gain experience with the complete machine learning workflow using scikit-learn.

## 🧠 Model and Implementation

| Component                  | Implementation                                                    | Libraries               |
| -------------------------- | ----------------------------------------------------------------- | ----------------------- |
| **Data loading**           | Read the Adult Income dataset from CSV                            | `pandas`                |
| **Missing value handling** | Replaced `?` with `NaN` and removed incomplete rows               | `pandas`, `numpy`       |
| **Target encoding**        | Mapped `<=50K` → 0 and `>50K` → 1                                 | `pandas`                |
| **Feature encoding**       | One-hot encoding using `pd.get_dummies()`                         | `pandas`                |
| **Train/test split**       | Stratified 80/20 split with fixed random seed                     | `scikit-learn`          |
| **Feature scaling**        | Standardized features using `StandardScaler`                      | `scikit-learn`          |
| **Model**                  | Support Vector Classifier with RBF kernel (`kernel='rbf'`, `C=1`) | `scikit-learn`          |
| **Training**               | Trained the SVM classifier using the training data                | `scikit-learn`          |
| **Prediction**             | Predicted income class on the test set                            | `scikit-learn`          |
| **Evaluation**             | Accuracy score, confusion matrix, and classification report       | `scikit-learn`          |
| **Visualization**          | Confusion matrix heatmap                                          | `matplotlib`, `seaborn` |

### Requirements

* Python 3.9+
* See `requirements.txt`
