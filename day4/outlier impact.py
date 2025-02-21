import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing

# Load dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Select feature (AveRooms) and target (MedHouseVal)
X = df[['AveRooms']]
y = data.target

# Detect outliers using Z-score method
df['Z_score'] = np.abs(zscore(X))  # Compute Z-score for the feature
outliers = df[df['Z_score'] > 3]    # Outliers: Z-score > 3
df_clean = df[df['Z_score'] <= 3]   # Clean dataset (without outliers)

# Split data (Original)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Split data (Without Outliers)
X_clean = df_clean[['AveRooms']]
y_clean = y[df_clean.index]
X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

# Train Linear Regression (Original Data)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2_original = r2_score(y_test, y_pred)

# Train Linear Regression (Without Outliers)
model_clean = LinearRegression()
model_clean.fit(X_train_clean, y_train_clean)
y_pred_clean = model_clean.predict(X_test_clean)
r2_clean = r2_score(y_test_clean, y_pred_clean)

# Print results
print(f'R² with Outliers: {r2_original:.4f}')
print(f'R² without Outliers: {r2_clean:.4f}')
print(f'Number of Outliers Removed: {len(outliers)}')

# Visualization
plt.scatter(X_test, y_test, color='blue', label='Original Data')
plt.scatter(X_test_clean, y_test_clean, color='green', label='Without Outliers', alpha=0.7)
plt.plot(X_test, y_pred, color='red', label='Regression Line (Original)')
plt.plot(X_test_clean, y_pred_clean, color='black', label='Regression Line (Without Outliers)')

plt.xlabel('Average Rooms per Household')
plt.ylabel('Median House Value')
plt.title('Outlier Impact on Regression')
plt.legend()
plt.show()
