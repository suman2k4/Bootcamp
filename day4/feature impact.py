import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing

# Load dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Use "AveRooms" (average rooms per household) as predictor and "MedHouseVal" as target
X = df[['AveRooms']]
y = data.target

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate with R-squared
r2 = r2_score(y_test, y_pred)
print(f'R-squared Score: {r2:.4f}')

# Plot regression line
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('Average Rooms per Household')
plt.ylabel('Median House Value')
plt.legend()
plt.show()

feature_importance['Absolute Impact'] = feature_importance['Coefficient'].abs()
feature_importance = feature_importance.sort_values(by='Absolute Impact', ascending=False)

# Print feature impact
print("Feature Impact Analysis:")
print(feature_importance[['Feature', 'Coefficient']])