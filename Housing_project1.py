# ==========================================
# HOUSING PRICE PREDICTOR - YOUR FIRST ML PROJECT
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

print("="*50)
print("🏠 HOUSING PRICE PREDICTOR")
print("="*50)

# ---- STEP 1: LOAD DATA ----
url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
df = pd.read_csv(url)

print(f"\n📊 Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head())

# ---- STEP 2: CHECK MISSING VALUES ----
print("\n" + "="*50)
print("🔍 MISSING VALUES")
print("="*50)
print(df.isnull().sum())

# ---- STEP 3: FILL MISSING VALUES ----
median_bedrooms = df['total_bedrooms'].median()
df['total_bedrooms'].fillna(median_bedrooms, inplace=True)
print(f"\n✅ Filled missing bedrooms with median: {median_bedrooms}")

# ---- STEP 4: CREATE NEW FEATURES ----
df['rooms_per_household'] = df['total_rooms'] / df['households']
df['bedrooms_per_household'] = df['total_bedrooms'] / df['households']
print("\n✅ Created new features: rooms_per_household, bedrooms_per_household")

# ---- STEP 5: VISUALIZE DATA ----
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Income vs House Value
axes[0, 0].scatter(df['median_income'], df['median_house_value'], alpha=0.5)
axes[0, 0].set_xlabel('Median Income')
axes[0, 0].set_ylabel('Median House Value')
axes[0, 0].set_title('Income vs House Value')

# Plot 2: Rooms per Household vs House Value
axes[0, 1].scatter(df['rooms_per_household'], df['median_house_value'], alpha=0.5)
axes[0, 1].set_xlabel('Rooms per Household')
axes[0, 1].set_ylabel('Median House Value')
axes[0, 1].set_title('Rooms per Household vs House Value')

# Plot 3: Distribution of House Values
axes[1, 0].hist(df['median_house_value'], bins=50, edgecolor='black')
axes[1, 0].set_xlabel('Median House Value')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Distribution of House Values')

# Plot 4: Bedrooms per Household
axes[1, 1].hist(df['bedrooms_per_household'].dropna(), bins=30, edgecolor='black')
axes[1, 1].set_xlabel('Bedrooms per Household')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Distribution of Bedrooms per Household')

plt.tight_layout()
plt.savefig('housing_plots.png')
print("\n✅ Plots saved as 'housing_plots.png'")
#---plt.show() ---

# ---- STEP 6: BUILD THE MODEL ----
X = df[['median_income', 'rooms_per_household']]
y = df['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n" + "="*50)
print("📈 MODEL PERFORMANCE")
print("="*50)
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred)):,.0f}")

# ---- STEP 7: SAVE THE MODEL ----
import joblib
joblib.dump(model, 'housing_model.pkl')
print("\n✅ Model saved as 'housing_model.pkl'")

# ---- STEP 8: PREDICT A SAMPLE HOUSE ----
sample_house = [[3.5, 5.5]]  # median_income, rooms_per_household
predicted_price = model.predict(sample_house)
print(f"\n🏠 Sample house (income=3.5, rooms/household=5.5) → Predicted price: ${predicted_price[0]:,.0f}")

print("\n" + "="*50)
print("🎉 PROJECT COMPLETE!")
print("="*50)
