# classification_algorithm_comparison
An Empirical Study of Classification Algorithms for Health Data

## Abstract

This study conducts an extensive empirical comparison of four machine learning algorithms—Support Vector Machines (SVM), Logistic Regression, Random Forest, and Artificial Neural Networks (ANN)—on three datasets sourced from the UCI Machine Learning Repository. The primary objective was to evaluate the performance of these models using multiple metrics, including accuracy, ROC AUC, and average precision-recall (APR). The results revealed that Logistic Regression consistently exceeded expectations, especially on smaller datasets, while Neural Networks exhibited significant underperformance. Overfitting was a recurring issue for SVM and Random Forest, especially on smaller datasets or those with limited features. These findings emphasize the critical importance of aligning model selection and configuration with dataset characteristics.

## 1. Introduction

Machine learning (ML) has become a cornerstone in predictive modeling and decision-making across diverse domains such as healthcare, finance, and engineering. Despite its growing adoption, the effectiveness of ML algorithms is highly dependent on factors such as dataset characteristics, model hyperparameters, and evaluation metrics. Given this variability, practitioners often face challenges in choosing the most suitable model for a specific problem.

This study seeks to provide insights into the behavior and performance of four widely used ML algorithms—SVM, Logistic Regression, Random Forest, and ANN—when applied to datasets of varying complexity and size. By systematically evaluating these models across multiple performance metrics, including accuracy, ROC AUC, and APR, we aim to identify their strengths, limitations, and suitability for specific types of data.

## 2. Methodology

### 2.1 Selected Algorithms

This study examined the following machine learning algorithms, with hyperparameters based on previous research (Caruana & Niculescu-Mizil, 2006) and our own parameter grid search:

- Support Vector Machines (SVM): Explored linear and non-linear kernels (polynomial, radial basis function). The regularization parameter C was varied from 10^-7 to 10^3, while the degree and gamma parameters were tuned for polynomial and RBF kernels.

- Logistic Regression: Trained with both unregularized and L2-regularized versions. The regularization parameter C was varied from 10^-8 to 10^4.

- Random Forest: Hyperparameters included n_estimators (100-500), feature selection range (1-20), and regularization settings (min_samples_split, min_samples_leaf) to control overfitting.

- Neural Networks (ANN): Multi-layer perceptron architectures with variations in hidden layers, activation functions ('relu', 'tanh'), solvers ('sgd', 'adam'), and learning rates.

### 2.2 Performance Metrics

The following evaluation metrics were used:

- Accuracy (ACC): Measures the proportion of correctly classified instances.

- ROC AUC: Evaluates the ability to distinguish between classes across all thresholds.

- Average Precision-Recall (APR): Useful for imbalanced datasets, emphasizing precision and recall trade-offs.

### 2.3 Data Sets

The datasets were selected from the UCI Machine Learning Repository:

- Haberman’s Survival Dataset: Small dataset with three features, challenging for complex models.

- Breast Cancer Wisconsin (Diagnostic) Dataset (WDBC): 30 real-valued features, larger size (569 instances).

- Early Stage Diabetes Risk Prediction Dataset: 16 attributes, mixed categorical and integer features, 520 instances.

## 3. Experiment

### 3.1 Hyper-Parameter Tuning

RandomizedSearchCV was employed for hyperparameter tuning due to computational efficiency. The function tune_model() was used for parameter selection, with n_iter = 10 and cv = 3.

### 3.2 Preliminary Training Results

Key findings include:

- SVM: High training accuracy but overfitting tendencies, especially on small datasets.

- Logistic Regression: Stable performance, minimal overfitting, good generalization.

- Random Forest: Strong accuracy but overfitting observed.

- Neural Network (ANN): Underperformed relative to other models.

### 3.3 Model-Specific Adjustments

- SVM: Increased regularization (C parameter) to mitigate overfitting.

- Logistic Regression: Applied Ridge regularization to improve generalization.

- Random Forest: Reduced max_depth and n_estimators to control overfitting.

- ANN: Used early stopping and learning rate adjustments, but improvements were limited.

### 3.4 Performance Metric Calculation

ROC curves and Precision-Recall Curves were generated using roc_curve and precision_recall_curve from Scikit-learn.

## 4. Discussion of Findings

- Random Forest: High accuracy but overfitting observed. Strong AUC and APR values.

- Logistic Regression: Stable performance, minimal overfitting, effective in distinguishing classes.

- SVM: Consistent accuracy, but slightly lower ROC AUC and APR compared to other models.

- Neural Networks (ANN): Underperformed, with lower accuracy and weak generalization.

## 5. Conclusion

This empirical study highlights the importance of selecting ML algorithms that align with dataset characteristics. Logistic Regression emerged as the most stable and reliable for smaller datasets, while Neural Networks struggled with generalization. Future work could explore additional feature engineering techniques and alternative neural architectures to enhance performance.

## 6. Usage Instructions

Dependencies

To reproduce the experiments, install the required dependencies:

pip install numpy pandas scikit-learn matplotlib seaborn

Running the Experiment

python run_experiment.py

Directory Structure

├── data/                                         # UCI datasets

├── figures/                                      # Output results

├── classification_algorithm_comparison.ipynb     # Main script

├── run_experiment.py                             # Script to run the experiment

└── README.md                                     # Project documentation

7. Future Improvements

- Expanding dataset selection to include multi-class classification problems.

- Experimenting with deep learning models beyond basic ANN architectures.

- Incorporating additional feature selection methods for improved performance.
