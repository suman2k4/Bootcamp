import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing

# Load dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Select feature (AveRooms) and target (MedHouseVal)
X = df[['AveRooms']]
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# Transform data for Polynomial Regression (degree=2)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train Polynomial Regression Model
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)

# Evaluate models
r2_linear = r2_score(y_test, y_pred_linear)
r2_poly = r2_score(y_test, y_pred_poly)

print(f'Linear Regression R²: {r2_linear:.4f}')
print(f'Polynomial Regression R²: {r2_poly:.4f}')

# Visualization
plt.scatter(X_test, y_test, color='blue', label='Actual Data')
plt.scatter(X_test, y_pred_linear, color='red', label='Linear Prediction', alpha=0.6)
plt.scatter(X_test, y_pred_poly, color='green', label='Polynomial Prediction', alpha=0.6)

plt.xlabel('Average Rooms per Household')
plt.ylabel('Median House Value')
plt.title('Linear vs Polynomial Regression')
plt.legend()
plt.show()
