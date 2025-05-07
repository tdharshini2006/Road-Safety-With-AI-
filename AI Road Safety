1. Install Required Libraries



import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00502/Metro_Interstate_Traffic_Volume.csv.gz'
df = pd.read_csv(url, compression='gzip')

# Preprocess data
df['date_time'] = pd.to_datetime(df['date_time'])
df['hour'] = df['date_time'].dt.hour
df['dayofweek'] = df['date_time'].dt.dayofweek

# Select features and target
features = ['hour', 'dayofweek', 'temp', 'rain_1h', 'snow_1h', 'clouds_all']
X = df[features]
y = df['traffic_volume']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Output
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Plot actual vs predicted
plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:100], label='Actual')
plt.plot(predictions[:100], label='Predicted')
plt.title('Traffic Volume Prediction (Sample)')
plt.xlabel('Sample')
plt.ylabel('Traffic Volume')
plt.legend()
plt.tight_layout()
plt.show()




Expected Output Example:

Mean Squared Error: 1189063.45
R² Score: 0.91
