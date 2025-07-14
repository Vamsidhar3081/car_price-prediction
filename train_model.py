import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

# Load your dataset
df = pd.read_csv(r"C:\Users\dandu\Desktop\projects\car project\cars.csv")

# Preprocessing
df['year'] = 2024 - df['year']              # Convert year to car age
df = df.rename(columns={'year': 'car_age'}) # Rename for clarity
df = df.drop(['name'], axis=1)              # Drop name column (string)

# Encode categorical features
df['fuel'] = df['fuel'].astype('category').cat.codes
df['seller_type'] = df['seller_type'].astype('category').cat.codes
df['transmission'] = df['transmission'].astype('category').cat.codes
df['owner'] = df['owner'].astype('category').cat.codes

# Define features and target
x = df.drop('selling_price', axis=1)
y = df['selling_price']

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor()
model.fit(x_train, y_train)

# Save model
with open("car_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as car_model.pkl")
