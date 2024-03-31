# -*- coding: utf-8 -*-
"""Keras Sequential Aware.ipynb

Automatically generated by Colaboratory.

"""

!pip install keras
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
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Separate features and target
X = final_df.drop(['id', 'p(Hallucination)_y'], axis=1)
y = final_df['p(Hallucination)_y']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.head(1))

!pip install keras-tuner
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from keras_tuner import RandomSearch

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from keras_tuner import RandomSearch

# Define the model building function required for Keras Tuner
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.InputLayer(input_shape=(X_train.shape[1],)))

    # Tune the number of layers from 1 to 3
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(layers.Dense(
            units=hp.Int(f'units_{i}', min_value=32, max_value=512, step=32),
            activation=hp.Choice(f'activation_{i}', ['relu', 'tanh', 'sigmoid']),
            kernel_regularizer=regularizers.l2(hp.Float('l2_reg', min_value=1e-5, max_value=1e-2, sampling='log'))
        ))
        # Tune the dropout rate
        model.add(layers.Dropout(rate=hp.Float(f'dropout_{i}', min_value=0.0, max_value=0.5, step=0.1)))

    # Output layer
    model.add(layers.Dense(1))  # Adjust according to your needs (e.g., classification vs regression)

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=hp.Choice('learning_rate', values=[1e-2, 2e-3, 5e-4])
        ),
        loss='mean_squared_error',
        metrics=['mean_squared_error']
    )

    return model

# Initialize the tuner
tuner = RandomSearch(
    build_model,
    objective='val_mean_squared_error',
    max_trials=200,
    executions_per_trial=2,
)

# Perform hyperparameter search
tuner.search(X_train, y_train, epochs=10, validation_split=0.2, verbose=2)

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete.
""")

model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train, y_train, validation_split=0.2, epochs=10000, verbose=2)  # Use a high epoch value

# Evaluate the model on the test set
eval_result = model.evaluate(X_test, y_test)
print("[test loss, test MSE]:", eval_result)

predictions = model.predict(X_test)
# Evaluation metrics
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R²): {r2}")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Discretizing the predictions using 0.5 as a cutoff
binary_predictions = np.where(predictions > 0.5, 1, 0)
binary_y_test = np.where(y_test > 0.5, 1, 0)

# Binary classification metrics
accuracy = accuracy_score(binary_y_test, binary_predictions)
precision = precision_score(binary_y_test, binary_predictions)
recall = recall_score(binary_y_test, binary_predictions)
f1 = f1_score(binary_y_test, binary_predictions)
conf_matrix = confusion_matrix(binary_y_test, binary_predictions)

# Print binary classification metrics
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'Confusion Matrix:\n{conf_matrix}')

import joblib
# Save the model
model_filename = 'best_sequential_aware.joblib'
joblib.dump(model, model_filename)
print(f"Model saved as {model_filename}")

from google.colab import files
files.download('best_sequential_aware.joblib')

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import json
best_gradient_model = joblib.load('best_sequential_aware.joblib')

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

y_pred = best_gradient_model.predict(final_df.drop('id', axis=1))
# y_pred = np.round(y_pred / 0.2) * 0.2
import numpy as np
import pandas as pd

result = []
for i in range(len(final_df)):
    id = str(final_df.loc[i, 'id'])
    p_hallucination = y_pred[i]
    label = 'Hallucination' if p_hallucination > 0.5 else 'Not Hallucination'
    result.append({"id": id, "p(Hallucination)": p_hallucination, "label": label})

def convert_array_to_float(result):
    for item in result:
        item['p(Hallucination)'] = float(item['p(Hallucination)'])
    return result

result = convert_array_to_float(result)

with open('test.model-aware.json', 'w') as f:
  json.dump(result, f)

result

