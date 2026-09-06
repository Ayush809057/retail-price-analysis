import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import joblib

# -----------------------------
# Load Dataset
# -----------------------------

current_folder = os.path.dirname(os.path.abspath(__file__))
csv_file = os.path.join(current_folder, "cafe_sales.csv")

data = pd.read_csv(csv_file)

print("Dataset")
print(data)

# -----------------------------
# Check Missing Values
# -----------------------------

print("\nMissing Values")
print(data.isnull().sum())

data.fillna(data.mean(numeric_only=True), inplace=True)

# -----------------------------
# Dataset Information
# -----------------------------

print("\nInformation")
print(data.info())

print("\nStatistics")
print(data.describe())

# -----------------------------
# Scatter Plot
# -----------------------------

plt.figure(figsize=(6,4))
plt.scatter(data["Price"], data["Quantity"])
plt.xlabel("Price")
plt.ylabel("Quantity")
plt.title("Price vs Quantity")
plt.show()

# -----------------------------
# Linear Regression
# -----------------------------

X = data[["Price"]]
y = data["Quantity"]

model = LinearRegression()
model.fit(X, y)

print("\nCoefficient:", model.coef_[0])
print("Intercept:", model.intercept_)

# -----------------------------
# Predictions
# -----------------------------

data["Predicted_Quantity"] = model.predict(X)

print("\nPrediction Table")
print(data)

# -----------------------------
# Elasticity
# -----------------------------

avg_price = data["Price"].mean()
avg_quantity = data["Quantity"].mean()

elasticity = model.coef_[0] * (avg_price / avg_quantity)

print("\nPrice Elasticity:", round(elasticity,3))

# -----------------------------
# Revenue Optimization
# -----------------------------

prices = np.arange(
    data["Price"].min(),
    data["Price"].max()+1
)

price_df = pd.DataFrame(prices, columns=["Price"])

predicted_quantity = model.predict(price_df)

revenue = prices * predicted_quantity

best = np.argmax(revenue)

print("\n========== BEST PRICE ==========")
print("Optimal Price :", prices[best])
print("Expected Quantity :", round(predicted_quantity[best],2))
print("Maximum Revenue :", round(revenue[best],2))

# -----------------------------
# Revenue Graph
# -----------------------------

plt.figure(figsize=(7,5))
plt.plot(prices, revenue)
plt.scatter(prices[best], revenue[best], s=120)
plt.xlabel("Price")
plt.ylabel("Revenue")
plt.title("Revenue Optimization")
plt.show()

# -----------------------------
# Save Model
# -----------------------------

model_file = os.path.join(current_folder, "price_model.pkl")

joblib.dump(model, model_file)

print("\nModel Saved Successfully")

# -----------------------------
# Load Model
# -----------------------------

loaded_model = joblib.load(model_file)

new_price = pd.DataFrame([[67]], columns=["Price"])

prediction = loaded_model.predict(new_price)

print(f"\nPredicted Quantity at Price 67 = {prediction[0]:.2f}")