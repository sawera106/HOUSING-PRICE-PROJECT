"""
============================================================
🏠 IMPROVED HOUSING PRICE PREDICTOR
Target: R² > 0.70
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

print("="*60)
print("🏠 IMPROVED HOUSING PRICE PREDICTOR")
print("="*60)

# ---- STEP 1: LOAD DATA ----
url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
df = pd.read_csv(url)

print(f"\n📊 Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ---- STEP 2: CLEAN DATA ----
# Fill missing values
median_bedrooms = df['total_bedrooms'].median()
df['total_bedrooms'].fillna(median_bedrooms, inplace=True)
print(f"✅ Filled {median_bedrooms} missing bedrooms")

# ---- STEP 3: FEATURE ENGINEERING (SMARTER FEATURES!) ----
print("\n🔧 Creating smart features...")

# Convert ocean_proximity to numbers (One-Hot Encoding)
df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)

# Ratio features (you already know these!)
df['rooms_per_household'] = df['total_rooms'] / df['households']
df['bedrooms_per_household'] = df['total_bedrooms'] / df['households']
df['people_per_household'] = df['population'] / df['households']

# Interaction features (new!)
df['income_rooms_interaction'] = df['median_income'] * df['rooms_per_household']
df['income_population_interaction'] = df['median_income'] * df['people_per_household']

# Geographic features (capture location effects)
df['latitude_squared'] = df['latitude'] ** 2
df['longitude_squared'] = df['longitude'] ** 2
df['lat_long_interaction'] = df['latitude'] * df['longitude']

print(f"✅ Created {df.shape[1]} total features")

# ---- STEP 4: REMOVE OUTLIERS ----
print("\n🧹 Removing outliers...")

# Remove houses with price > 95th percentile (extreme outliers)
price_upper = df['median_house_value'].quantile(0.95)
df = df[df['median_house_value'] <= price_upper]

# Remove houses with rooms_per_household > 95th percentile
rooms_upper = df['rooms_per_household'].quantile(0.95)
df = df[df['rooms_per_household'] <= rooms_upper]

print(f"✅ Removed outliers. Remaining rows: {df.shape[0]}")

# ---- STEP 5: SEPARATE FEATURES AND TARGET ----
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

print(f"\n📋 Features: {X.shape[1]}")
print(f"📊 Target: median_house_value")

# ---- STEP 6: TRAIN/TEST SPLIT ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---- STEP 7: SCALE FEATURES ----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---- STEP 8: TRAIN MODELS ----
print("\n" + "="*60)
print("📈 TRAINING MODELS")
print("="*60)

# Model 1: Linear Regression
print("\n🔹 Linear Regression...")
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
lr_r2 = r2_score(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
print(f"   R²: {lr_r2:.4f}")
print(f"   RMSE: ${lr_rmse:,.0f}")

# Model 2: Random Forest (STRONGER!)
print("\n🔹 Random Forest (n_estimators=100)...")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
rf_pred = rf.predict(X_test_scaled)
rf_r2 = r2_score(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
print(f"   R²: {rf_r2:.4f}")
print(f"   RMSE: ${rf_rmse:,.0f}")

# Model 3: Random Forest (MORE TREES!)
print("\n🔹 Random Forest (n_estimators=200)...")
rf2 = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf2.fit(X_train_scaled, y_train)
rf2_pred = rf2.predict(X_test_scaled)
rf2_r2 = r2_score(y_test, rf2_pred)
rf2_rmse = np.sqrt(mean_squared_error(y_test, rf2_pred))
print(f"   R²: {rf2_r2:.4f}")
print(f"   RMSE: ${rf2_rmse:,.0f}")

# ---- STEP 9: PICK BEST MODEL ----
models = {
    'Linear Regression': (lr, lr_r2, lr_rmse),
    'Random Forest (100)': (rf, rf_r2, rf_rmse),
    'Random Forest (200)': (rf2, rf2_r2, rf2_rmse)
}

best_model_name = max(models, key=lambda x: models[x][1])
best_model, best_r2, best_rmse = models[best_model_name]

print("\n" + "="*60)
print("🏆 BEST MODEL")
print("="*60)
print(f"Model:   {best_model_name}")
print(f"R²:      {best_r2:.4f}")
print(f"RMSE:    ${best_rmse:,.0f}")
print("="*60)

# ---- STEP 10: SAVE THE BEST MODEL ----
joblib.dump(best_model, 'housing_model_improved.pkl')
joblib.dump(scaler, 'scaler_improved.pkl')
print("✅ Best model saved as 'housing_model_improved.pkl'")
print("✅ Scaler saved as 'scaler_improved.pkl'")

# ---- STEP 11: FEATURE IMPORTANCE (What matters most?) ----
if hasattr(best_model, 'feature_importances_'):
    print("\n🔍 TOP 5 MOST IMPORTANT FEATURES:")
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False).head(5)
    
    for i, row in feature_importance.iterrows():
        print(f"   {row['Feature']}: {row['Importance']:.3f}")

# ---- STEP 12: VISUALIZE IMPROVEMENT ----
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, lr_pred, alpha=0.3, label='Linear Regression')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title(f'Linear Regression (R²={lr_r2:.3f})')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_test, rf_pred, alpha=0.3, color='green', label='Random Forest')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title(f'Random Forest (R²={rf_r2:.3f})')
plt.legend()

plt.tight_layout()
plt.savefig('improved_housing_plots.png', dpi=150)
print("✅ Improved plots saved as 'improved_housing_plots.png'")
plt.show()

# ---- STEP 13: INTERACTIVE PREDICTOR ----
print("\n" + "="*60)
print("🏠 INTERACTIVE PRICE PREDICTOR (Improved)")
print("="*60)
print("Enter house details to get a predicted price.\n")

print("📋 Note: Enter ALL features to get the best prediction!")
print("   Type 'quit' to exit.\n")

# Sample default values
sample = X.iloc[0].to_dict()

while True:
    try:
        user_input = {}
        print("\nEnter feature values (press Enter to use default):")
        for col in X.columns[:10]:  # Show first 10 features (shorter list)
            default = sample.get(col, 0)
            val = input(f"  {col} [default: {default:.2f}]: ")
            if val.lower() == 'quit':
                print("\n👋 Goodbye!")
                break
            if val == '':
                user_input[col] = default
            else:
                user_input[col] = float(val)
        else:
            # Predict
            input_df = pd.DataFrame([user_input])
            input_scaled = scaler.transform(input_df)
            prediction = best_model.predict(input_scaled)[0]
            print(f"\n💰 Predicted house price: ${prediction:,.0f}")
            print("="*50)
            continue
        break
    except ValueError:
        print("❌ Please enter a valid number!\n")
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        break

print("\n" + "="*60)
print("🎉 IMPROVED PROJECT COMPLETE!")
print("="*60)
print(f"✅ Best R²: {best_r2:.4f} (up from 0.4642)")
print(f"✅ Best RMSE: ${best_rmse:,.0f} (improvement from ${lr_rmse:,.0f})")
print("="*60)
