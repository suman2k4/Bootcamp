import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing

# Load dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Select multiple features
X = df[['AveRooms', 'Population', 'HouseAge', 'MedInc']]
y = data.target  # Target variable (Median House Value)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Standard Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
r2_linear = r2_score(y_test, y_pred_linear)

# Train Ridge Regression Model
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)
r2_ridge = r2_score(y_test, y_pred_ridge)

# Train Lasso Regression Model
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)
r2_lasso = r2_score(y_test, y_pred_lasso)

# Print results
print(f'Linear Regression R²: {r2_linear:.4f}')
print(f'Ridge Regression R²: {r2_ridge:.4f}')
print(f'Lasso Regression R²: {r2_lasso:.4f}')

# Visualization
plt.figure(figsize=(6, 4))
plt.bar(['Linear', 'Ridge', 'Lasso'], [r2_linear, r2_ridge, r2_lasso], color=['blue', 'red', 'green'])
plt.ylabel('R-squared Score')
plt.title('Regularization Impact on Model Performance')
plt.show()
