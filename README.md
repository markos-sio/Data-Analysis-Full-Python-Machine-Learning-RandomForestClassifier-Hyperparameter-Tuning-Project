Medical Appointment No-Show Analysis
This project analyzes medical appointment no-show data to understand the factors influencing patient attendance. It includes data preprocessing, exploratory data analysis, and predictive modeling using logistic regression and random forest classifiers with SMOTE to address class imbalance.


Project Overview
1)Data Loading and Cleaning

Import CSV file and create a DataFrame.
Rename columns for better readability.
Check for missing and duplicate values.
Convert date columns to datetime format.
Drop irrelevant columns (PatientId and AppointmentID).
Handle incorrect age values.
Feature Engineering

2)Calculate waiting days between scheduling and appointment.

Encode categorical columns (Gender and Show).
One-hot encode the Handicap column.
Extract useful features from datetime columns.
Exploratory Data Analysis

3)Generate plots to visualize the distribution of waiting days and other columns of interest.

Create correlation matrix heatmap to understand feature relationships.

4)Predictive Modeling

Logistic Regression
Perform K-Fold cross-validation.
Evaluate model performance using confusion matrix, classification report, and accuracy score.
Random Forest with SMOTE
Address class imbalance using SMOTE.
Evaluate model performance with K-Fold cross-validation.
Hyperparameter Tuning
Use RandomizedSearchCV to find the best parameters for the Random Forest model.
Evaluate the tuned model's performance.