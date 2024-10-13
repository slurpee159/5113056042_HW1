import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Streamlit App for Linear Regression

# 1. Business Understanding
st.title('Linear Regression Analysis Using CRISP-DM Steps')
st.write("""
This app allows you to explore the impact of a variable "slope" on a linear regression model.
You can adjust the slope, noise scale, and the number of data points to see how they affect the regression line and model performance.
""")

# 2. Data Understanding
st.sidebar.header('Data Parameters')
slope_a = st.sidebar.slider('Slope (a)', min_value=-100, max_value=100, value=1)
noise_scale_c = st.sidebar.slider('Noise scale (c)', min_value=0, max_value=100, value=10)
num_points = st.sidebar.slider('Number of points (n)', min_value=10, max_value=500, value=100)

# Function to generate synthetic data
def generate_data(a, c, n):
    X = np.linspace(-10, 10, n)
    noise = np.random.normal(0, 1, n)
    y = a * X + 50 + c * noise
    return pd.DataFrame({'X': X, 'y': y})

# Generate data
data = generate_data(slope_a, noise_scale_c, num_points)

# 3. Data Preparation
st.write("## Dataset")
st.write(data.head())

# 4. Modeling
X = data['X'].values.reshape(-1, 1)
y = data['y'].values

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# 5. Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"### Model Performance")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"R-squared (R²): {r2:.2f}")

# 6. Deployment
st.write("## Visualizing the Results")

# Scatter plot of actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_pred, color='green', label='Predicted')
plt.plot(X_test, y_pred, color='red', label='Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Actual vs Predicted Values with Regression Line')
plt.legend()
st.pyplot(plt)

# 7. Monitoring and Maintenance
st.sidebar.write("Adjust the parameters to rerun the analysis with different configurations.")