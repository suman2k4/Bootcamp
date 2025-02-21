import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing

# Load dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Use "AveRooms" as predictor and "MedHouseVal" as target
X = df[['AveRooms']]
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Visualization using Seaborn
plt.figure(figsize=(8, 5))
sns.scatterplot(x=X_test.values.flatten(), y=y_test, color='blue', label='Actual Data')
sns.lineplot(x=X_test.values.flatten(), y=y_pred, color='red', label='Regression Line')

# Add labels and title
plt.xlabel('Average Rooms per Household')
plt.ylabel('Median House Value')
plt.title('Linear Regression: House Prices vs. Rooms')
plt.legend()
plt.show()
