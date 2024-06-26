# -*- coding: utf-8 -*-
"""Random Forest Model Aware.ipynb

Automatically generated by Colaboratory.

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import json

# Load feature files
feature_files = [f'/content/model-aware-500-{i}.json' for i in range(1, 9)]
feature_dfs = [pd.read_json(file) for file in feature_files]

# Merge feature files into a single DataFrame
merged_features_df = pd.concat(feature_dfs, ignore_index=True)

# Load target file
target_file = 'val.500.model-aware.json'
target_df = pd.read_json(target_file)

# Merge features with target on 'id'
merged_df = pd.merge(merged_features_df, target_df[['id', 'p(Hallucination)']], on='id')

merged_df.head(20)

# Preprocessing

# Add a new column for the sequence
merged_df['sequence'] = merged_df.groupby('id').cumcount() + 1

# Pivot the DataFrame
pivoted_df = merged_df.pivot(index='id', columns='sequence', values='p(Hallucination)_x')

# Rename the columns
pivoted_df.columns = [f'p(Hallucination)_{i}' for i in range(1, len(pivoted_df.columns)+1)]

# Reset the index
pivoted_df.reset_index(inplace=True)

# Merge the pivoted DataFrame with the original DataFrame to get the 'p(Hallucination)_y' column
final_df = pd.merge(pivoted_df, merged_df[['id', 'p(Hallucination)_y']].drop_duplicates(), on='id')

# Display the DataFrame
pd.set_option('display.max_columns', None)

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import json

# Splitting data into training and testing sets
X = final_df.drop('p(Hallucination)_y', axis=1)
y = final_df['p(Hallucination)_y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model selection and training
model = RandomForestRegressor(random_state=42)

# Define the grid of hyperparameters to search
hyperparameter_grid = {
    'n_estimators': [100, 250, 500, 1000],
    'max_depth': [5, 15, 30],
    'min_samples_split': [2, 10, 30, 100],
    'min_samples_leaf': [2, 5, 10, 25],
    'max_features': ['auto', 'sqrt'],
    'bootstrap': [True, False]
}

# Set up the grid search
grid_cv = GridSearchCV(estimator=model,
                       param_grid=hyperparameter_grid,
                       cv=6,
                       n_jobs = -1,
                       verbose=2)

# Fit the grid search model
grid_cv.fit(X_train, y_train)

# Get the best parameters
best_params = grid_cv.best_params_
print(f'Best parameters: {best_params}')

# Train the model using the best parameters
best_model = RandomForestRegressor(n_estimators=best_params['n_estimators'],
                                   max_depth=best_params['max_depth'],
                                   min_samples_split=best_params['min_samples_split'],
                                   min_samples_leaf=best_params['min_samples_leaf'],
                                   max_features=best_params['max_features'],
                                   bootstrap=best_params['bootstrap'],
                                   random_state=42)
# best_model = RandomForestRegressor()
best_model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np

# Model evaluation with additional metrics
y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Convert predictions and actual values to binary classification
y_pred_binary = [1 if x >= 0.5 else 0 for x in y_pred]
y_test_binary = [1 if x >= 0.5 else 0 for x in y_test]

# Calculate binary classification metrics
accuracy = accuracy_score(y_test_binary, y_pred_binary)
precision = precision_score(y_test_binary, y_pred_binary)
recall = recall_score(y_test_binary, y_pred_binary)
f1 = f1_score(y_test_binary, y_pred_binary)
conf_matrix = confusion_matrix(y_test_binary, y_pred_binary)

# Print binary classification metrics
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'Confusion Matrix:\n{conf_matrix}')

import joblib
# Save the model
model_filename = 'best_random_forest_model.joblib'
joblib.dump(best_model, model_filename)
print(f"Model saved as {model_filename}")

from google.colab import files
files.download('best_random_forest_model(1).joblib')

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import json
best_random_forest_model = joblib.load('best_random_forest_model(1).joblib')

#Load feature files
feature_files = [f'/content/model-aware-1500-{i}.json' for i in range(1, 9)]
feature_dfs = [pd.read_json(file) for file in feature_files]

# Merge feature files into a single DataFrame
merged_features_df = pd.concat(feature_dfs, ignore_index=True)

merged_features_df.sort_values('id').head(20)

# Preprocessing

# Group by 'id' and get the list of p(Hallucination) for each group
grouped = merged_features_df.groupby('id')['p(Hallucination)'].apply(list)

# Create new dataframe from the grouped data
new_df = pd.DataFrame(grouped)

# Create new columns p(Hallucination)_1, p(Hallucination)_2, ...
new_df = pd.DataFrame(new_df['p(Hallucination)'].to_list(), index=new_df.index)

# Rename the columns
new_df.columns = [f'p(Hallucination)_{i+1}' for i in range(len(new_df.columns))]

# Reset the index
new_df.reset_index(inplace=True)

# Display the DataFrame
pd.set_option('display.max_columns', None)
final_df=new_df
final_df.head(20)

y_pred = best_random_forest_model.predict(final_df)
import numpy as np
import pandas as pd

result = []
for i in range(len(final_df)):
    id = str(final_df.loc[i, 'id'])
    p_hallucination = y_pred[i]
    label = 'Hallucination' if p_hallucination > 0.5 else 'Not Hallucination'
    result.append({"id": id, "p(Hallucination)": p_hallucination, "label": label})

with open('test.model-aware.json', 'w') as f:
    json.dump(result, f)

import matplotlib.pyplot as plt


# Assuming 'best_random_forest_model' is your Random Forest model
feature_importances = best_random_forest_model.feature_importances_

# Adjusting feature names and importances according to your specifications
# Dropping 'id' feature is assumed to be handled prior, and features are renumbered accordingly

# Summing feature importances for features 1-3 and 4-6 (assuming 0-based indexing)
combined_importances = np.array([
    np.sum(feature_importances[0:3])/np.sum(feature_importances[0:3]+feature_importances[7]+feature_importances[8]), # Sum importances for features 1-3
    np.sum(feature_importances[7])/np.sum(feature_importances[0:3]+feature_importances[7]+feature_importances[8]),
    np.sum(feature_importances[8])/np.sum(feature_importances[0:3]+feature_importances[7]+feature_importances[8])
])
combined_features = ['GPT', 'SelfCheckGPT', 'Vectara']

# Plotting
plt.bar(combined_features, combined_importances)
plt.xlabel('Feature Groups')
plt.ylabel('Importance')
plt.title('Combined Feature Importance Plot')
plt.xticks(rotation='vertical')
plt.show()

from sklearn.tree import plot_tree

# Visualizing the first tree from the Random Forest
plt.figure(figsize=(20,10))
plot_tree(best_random_forest_model.estimators_[0],
          feature_names=final_df.columns,
          filled=True, impurity=True,
          rounded=True)
plt.show()

!pip install --upgrade scikit-learn

from sklearn.inspection import plot_partial_dependence

fig, ax = plt.subplots(figsize=(10, 6))
plot_partial_dependence(best_random_forest_model, features=[0, 1], X=final_df, ax=ax) # for the first two features
plt.show()

import json

# First, ensure that the file 'test.model-aware.json' is uploaded to the correct directory
# Assuming the file is in the current working directory or a specified path

file_path = 'test.model-aware.json' # Update this path if the file is in a different directory

try:
    with open(file_path, 'r') as file:
        y_test = json.load(file)
    print("File loaded successfully!")
    # If you want to print or check the first few entries of y_test:
    # print(y_test[:5])  # Assuming y_test is a list; adjust this line if y_test has a different structure
except FileNotFoundError:
    print(f"File not found: {file_path}")
except json.JSONDecodeError:
    print(f"Error decoding JSON from the file: {file_path}")

y_test = np.array([item['p(Hallucination)'] for item in y_test])

from sklearn.metrics import mean_absolute_error

predictions = best_random_forest_model.predict(final_df) # Assuming X_test is your test dataset
plt.scatter(y_test, predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.axis('square')
plt.plot([-100, 100], [-100, 100]) # Adjust limits accordingly
plt.show()

residuals = y_test - y_pred
plt.scatter(predictions, residuals)
plt.xlabel('Predictions')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='-')
plt.show()







