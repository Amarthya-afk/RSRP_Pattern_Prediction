import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score

# Load and preprocess the data
file = 'op2_downsampled.csv'
base = pd.read_csv(file, header=None)
base.columns = ['Time', 'SSS-RSRP', 'SSS-RePower']
base = base.iloc[1:]  # Remove header row
base['SSS-RSRP'] = base['SSS-RSRP'].astype(float)

# Smooth RSRP values with moving average
base['SSS-RSRP-smoothed'] = base['SSS-RSRP'].rolling(window=5).mean()
base = base.dropna()
myrsrp = base['SSS-RSRP-smoothed'].head(10000).values.reshape(-1, 1)

# Normalize``
scaler = MinMaxScaler(feature_range=(0, 1))
myrsrp_norm = scaler.fit_transform(myrsrp)

# Split into training and testing sets
rsrptrain = myrsrp_norm[:9900, :]
rsrptest = myrsrp_norm[9900:, :]


# Create training sequences (150-step)
window_size = 150
prev = []
real_rsrp = []
for i in range(window_size, rsrptrain.shape[0]):
    prev.append(rsrptrain[i-window_size:i, 0])
    real_rsrp.append(rsrptrain[i, 0])
prev, real_rsrp = np.array(prev), np.array(real_rsrp)
prev = np.reshape(prev, (prev.shape[0], prev.shape[1], 1))

# Build LSTM model
regressor = Sequential()
regressor.add(LSTM(units=64, return_sequences=True, input_shape=(prev.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=32))
regressor.add(Dropout(0.2))
regressor.add(Dense(units=1))

regressor.compile(optimizer='adam', loss='mae', metrics=['mse'])
history = regressor.fit(prev, real_rsrp, epochs=80, batch_size=32, validation_split=0.1)

# Prepare test input
inputs = myrsrp_norm[len(myrsrp_norm) - len(rsrptest) - window_size:]
x_test = []
for i in range(window_size, len(inputs)):
    x_test.append(inputs[i-window_size:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Predict
prediction = regressor.predict(x_test)
prediction = scaler.inverse_transform(prediction)
real_rsrp_test = myrsrp[9900:, :]
# Smooth prediction to show trend
smoothed_pred = pd.Series(prediction.flatten()).rolling(window=5).mean().to_numpy()


# Plotting
plt.figure(figsize=(14, 6))
plt.plot(real_rsrp_test, color='blue', label='Actual SSS-RSRP')
plt.plot(prediction, color='red', linestyle='--', label='Raw Prediction')
plt.plot(smoothed_pred, color='green', linewidth=2, label='Smoothed Prediction')
plt.fill_between(range(len(real_rsrp_test)), real_rsrp_test.flatten(), prediction.flatten(), color='gray', alpha=0.2, label='Prediction Error')
plt.title("RSRP Trend Prediction (9900 training samples)")
plt.xlabel("Sample Index")
plt.ylabel("SSS-RSRP (dBm)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot training loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training Loss (MAE)')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()


# Flatten arrays
y_true = real_rsrp_test.flatten()
y_pred = prediction.flatten()

# MAE (already implicitly used)
mae = mean_absolute_error(y_true, y_pred)

# MSE
mse = mean_squared_error(y_true, y_pred)

# RMSE
rmse = np.sqrt(mse)

# R2 Score
r2 = r2_score(y_true, y_pred)

print("MAE  :", mae)
print("MSE  :", mse)
print("RMSE :", rmse)
print("R²   :", r2)
