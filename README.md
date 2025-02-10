# Featurization-Model-Selection-and-Tuning

## Semiconductor Manufacturing Process - Yield Prediction

### Project Overview
This project focuses on analyzing and building a machine learning classifier to predict the Pass/Fail yield of semiconductor manufacturing process entities. The goal is to determine whether all features are required for the model or if feature selection can improve performance.

### Domain: Semiconductor Manufacturing Process

### Context:
Modern semiconductor manufacturing involves constant monitoring of signals/variables collected from sensors and measurement points. These signals contain useful information, irrelevant data, and noise. Engineers often deal with an overwhelming number of signals, many of which are unnecessary. By applying feature selection, we aim to identify the most relevant signals that impact yield type, enabling:
Increased process throughput
Reduced time to learning
Lower per-unit production costs

### Data Description:
Dataset: sensor-data.csv
Shape: 1567 datapoints with 591 features (excluding the target column)
Target Variable:
-1: Pass
1: Fail
Each row represents a single production entity with associated measured features.

### Project Workflow
1. Data Import and Understanding
Load the dataset and explore its basic structure.
Perform a 5-point summary to gain insights into the data distribution and quality.
2. Data Cleansing
Remove features with more than 20% missing values.
Impute remaining missing values with the mean.
Identify and drop features with constant values across all rows.
Address multicollinearity by analyzing correlations among features.
Apply logical/functional reasoning to remove irrelevant features.
3. Data Analysis & Visualization
Conduct univariate analysis for feature distributions.
Perform bivariate and multivariate analysis to explore relationships between features and the target variable.
4. Data Preprocessing
Separate predictors (features) and the target variable.
Check for class imbalance in the target variable and apply balancing techniques if necessary.
Perform train-test split and standardize/normalize data as required.
Compare statistical characteristics of train/test splits with the original dataset.
5. Model Training, Testing, and Tuning
Train models using supervised learning techniques such as Logistic Regression, SVM, Random Forest, etc.
Use cross-validation to evaluate model performance.
Apply hyperparameter tuning (e.g., Grid Search) to optimize model accuracy.
Experiment with techniques like dimensionality reduction, feature elimination, or balancing to enhance performance.
Generate a classification report detailing precision, recall, F1-score, etc.
6. Post-Training Analysis
Compare all trained models based on their train/test accuracies.
Select the best-performing model with justification.
Save (pickle) the final model for future use.
Write conclusions summarizing insights and results.

### Tools & Libraries
Programming Language: Python
Libraries:
Data Handling: Pandas, NumPy
Visualization: Matplotlib, Seaborn
Machine Learning: Scikit-learn
Model Saving: Pickle

### Expected Outcomes
A robust classifier capable of predicting Pass/Fail outcomes accurately.
Insights into which features significantly impact yield type.
Optimized feature set for efficient modeling and reduced computational overhead.
