import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Read the fighters dataset from CSV
fighters_df = pd.read_csv('fighters.csv')

# Read the popular matches dataset from CSV
matches_df = pd.read_csv('popular_matches.csv')

# DATA Cleaning
# Ensure 'ko_rate' is treated as a string before stripping the '%' character
fighters_df['ko_rate'] = fighters_df['ko_rate'].astype(str).str.rstrip('%').astype(float) / 100.0

# Handling outliers and inconsistencies for Fighters Dataset:
fighters_df['age'] = pd.to_numeric(fighters_df['age'], errors='coerce')
fighters_df.loc[fighters_df['age'] < 0, 'age'] = pd.NA

# Data Cleaning Tasks for Popular Matches Dataset:
matches_df.replace('Unknown', pd.NA, inplace=True)
numerical_cols = ['opponent_1_estimated_punch_power', 'opponent_2_estimated_punch_power',
                  'opponent_1_estimated_punch_resistance', 'opponent_2_estimated_punch_resistance',
                  'opponent_1_estimated_ability_to_take_punch', 'opponent_2_estimated_ability_to_take_punch',
                  'opponent_1_rounds_boxed', 'opponent_2_rounds_boxed',
                  'opponent_1_round_ko_percentage', 'opponent_2_round_ko_percentage',
                  'opponent_1_has_been_ko_percentage', 'opponent_2_has_been_ko_percentage',
                  'opponent_1_avg_weight', 'opponent_2_avg_weight']
matches_df[numerical_cols] = matches_df[numerical_cols].apply(pd.to_numeric, errors='coerce')
matches_df.loc[matches_df['opponent_1_rounds_boxed'] < 0, 'opponent_1_rounds_boxed'] = pd.NA

# Create target variable based on the revised logic
def determine_winner(verdict):
    if 'won' in verdict.lower():
        if 'via' in verdict.lower():
            result_part = verdict.lower().split('via')[0]
            if 'won' in result_part:
                return 1  # assuming opponent_1 is the winner if 'won' appears before 'via'
        else:
            return 1
    return 0

matches_df['winner'] = matches_df['verdict'].apply(determine_winner)

# Display the cleaned datasets
fighters_df.head()
matches_df.head()

# EDA - Skipped for brevity

# Feature Engineering: Difference and Ratio Features
matches_df['punch_power_diff'] = matches_df['opponent_1_estimated_punch_power'] - matches_df['opponent_2_estimated_punch_power']
matches_df['punch_resistance_diff'] = matches_df['opponent_1_estimated_punch_resistance'] - matches_df['opponent_2_estimated_punch_resistance']
matches_df['ability_to_take_punch_diff'] = matches_df['opponent_1_estimated_ability_to_take_punch'] - matches_df['opponent_2_estimated_ability_to_take_punch']
matches_df['rounds_boxed_diff'] = matches_df['opponent_1_rounds_boxed'] - matches_df['opponent_2_rounds_boxed']
matches_df['round_ko_percentage_diff'] = matches_df['opponent_1_round_ko_percentage'] - matches_df['opponent_2_round_ko_percentage']
matches_df['has_been_ko_percentage_diff'] = matches_df['opponent_1_has_been_ko_percentage'] - matches_df['opponent_2_has_been_ko_percentage']
matches_df['avg_weight_diff'] = matches_df['opponent_1_avg_weight'] - matches_df['opponent_2_avg_weight']

matches_df['punch_power_ratio'] = matches_df['opponent_1_estimated_punch_power'] / matches_df['opponent_2_estimated_punch_power']
matches_df['punch_resistance_ratio'] = matches_df['opponent_1_estimated_punch_resistance'] / matches_df['opponent_2_estimated_punch_resistance']
matches_df['ability_to_take_punch_ratio'] = matches_df['opponent_1_estimated_ability_to_take_punch'] / matches_df['opponent_2_estimated_ability_to_take_punch']
matches_df['rounds_boxed_ratio'] = matches_df['opponent_1_rounds_boxed'] / matches_df['opponent_2_rounds_boxed']
matches_df['round_ko_percentage_ratio'] = matches_df['opponent_1_round_ko_percentage'] / matches_df['opponent_2_round_ko_percentage']
matches_df['has_been_ko_percentage_ratio'] = matches_df['opponent_1_has_been_ko_percentage'] / matches_df['opponent_2_has_been_ko_percentage']
matches_df['avg_weight_ratio'] = matches_df['opponent_1_avg_weight'] / matches_df['opponent_2_avg_weight']

# Interaction Features
matches_df['punch_power_x_punch_resistance_1'] = matches_df['opponent_1_estimated_punch_power'] * matches_df['opponent_1_estimated_punch_resistance']
matches_df['punch_power_x_punch_resistance_2'] = matches_df['opponent_2_estimated_punch_power'] * matches_df['opponent_2_estimated_punch_resistance']

# Handling NaN values in newly created features
matches_df.fillna(0, inplace=True)
matches_df.head()

# Select relevant features and target variable
features = ['punch_power_diff', 'punch_resistance_diff', 'ability_to_take_punch_diff', 'rounds_boxed_diff',
            'round_ko_percentage_diff', 'has_been_ko_percentage_diff', 'avg_weight_diff', 'punch_power_ratio',
            'punch_resistance_ratio', 'ability_to_take_punch_ratio', 'rounds_boxed_ratio',
            'round_ko_percentage_ratio', 'has_been_ko_percentage_ratio', 'avg_weight_ratio',
            'punch_power_x_punch_resistance_1', 'punch_power_x_punch_resistance_2']

X = matches_df[features]
y = matches_df['winner']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)

# Define a custom function to check if the model is fitted
def check_fitted(model):
    try:
        # Check if the model has any coefficients, indicating it's fitted
        if model.coef_.any():
            return True
        else:
            return False
    except:
        return False

# Fit the model if it's not already fitted
if not check_fitted(lr_model):
    lr_model.fit(X_train, y_train)

# Now the model should be fitted, proceed with predictions
lr_predictions = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_predictions)
print('Logistic Regression Accuracy:', lr_accuracy)

# Print confusion matrix and classification report
print('Confusion Matrix:')
print(confusion_matrix(y_test, lr_predictions))
print('Classification Report:')
print(classification_report(y_test, lr_predictions))

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print('Random Forest Accuracy:', rf_accuracy)
print(confusion_matrix(y_test, rf_predictions))
print(classification_report(y_test, rf_predictions))

# Support Vector Machine
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)
print('SVM Accuracy:', svm_accuracy)
print(confusion_matrix(y_test, svm_predictions))
print(classification_report(y_test, svm_predictions))

# Compare the performance of different models and select the best-performing one
models = ['Logistic Regression', 'Random Forest', 'SVM']
accuracies = [lr_accuracy, rf_accuracy, svm_accuracy]

best_model_index = np.argmax(accuracies)
print(f'The best-performing model is {models[best_model_index]} with an accuracy of {accuracies[best_model_index]:.2f}')

# Grid Search for Logistic Regression
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2']
}

lr_model = LogisticRegression(random_state=42, solver='liblinear')
grid_search = GridSearchCV(lr_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_
print("Best Parameters:", best_params)
print("Best Score (accuracy):", best_score)

best_model = grid_search.best_estimator_
best_model_predictions = best_model.predict(X_test)
best_model_accuracy = accuracy_score(y_test, best_model_predictions)
print('Best Model Accuracy:', best_model_accuracy)
print(confusion_matrix(y_test, best_model_predictions))
print(classification_report(y_test, best_model_predictions))


#Deployment
import joblib
file='winner'
joblib.dump(rf_model,"boxer")
model=joblib.load(open("boxer",'rb'))