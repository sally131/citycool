from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

# Sample dataset: Replace this with your actual dataset
data = {
    'avg_temp': [31, 33, 35, 30, 32, 29, 37, 36],
    'population_density': [1000, 1200, 1100, 950, 980, 870, 1300, 1250],
    'CO2_levels': [50, 55, 52, 45, 48, 40, 60, 58],
    'green_coverage': [20, 15, 12, 22, 19, 24, 8, 10],
    'heat_index': [70, 75, 76, 65, 68, 63, 80, 78]
}

df = pd.DataFrame(data)

# Features (X) and target variable (y)
X = df[['avg_temp', 'population_density', 'CO2_levels', 'green_coverage']]
y = df['heat_index']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor model
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model to the training data
rf_regressor.fit(X_train, y_train)

# Predict the heat index for the test set
y_pred = rf_regressor.predict(X_test)

# Calculate the mean squared error of the predictions
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Sample new data to predict
new_data = np.array([[33, 1100, 53, 15],
                     [29, 900, 45, 22]])

# Make predictions on new data
new_predictions = rf_regressor.predict(new_data)
print(f'Predicted heat indices for new data: {new_predictions}')
