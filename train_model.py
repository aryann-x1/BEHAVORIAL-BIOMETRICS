import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
import joblib

# Load the dataset
data = pd.read_csv('DSL-StrongPasswordData.csv')

# Step 1: Feature Extraction (Calculating time differences)
data['time_diff'] = data['H.t'].diff().fillna(0)

# Step 2: Prepare sequences (you can adjust the sequence length)
sequence_length = 10

# Group the data by subject and create sequences of time differences
sequences = []
labels = []
target_subject = 's033'  # Target user

for subject in data['subject'].unique():
    user_data = data[data['subject'] == subject]['time_diff'].values
    for i in range(len(user_data) - sequence_length):
        sequences.append(user_data[i:i + sequence_length])
        labels.append(1 if subject == target_subject else 0)

# Convert to NumPy arrays
X = np.array(sequences)
y = np.array(labels)

# Reshape X to be 3D [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(sequence_length, 1)))  # LSTM with 64 units
model.add(Dense(1, activation='sigmoid'))  # Output layer with sigmoid for binary classification

# Step 5: Compile and train the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Step 6: Evaluate the model (optional)
accuracy = model.evaluate(X_test, y_test)[1]
print(f"LSTM Model Accuracy: {accuracy * 100:.2f}%")

# Step 7: Save the model
model.save('auth_lstm_model.h5')
print("LSTM model saved as 'auth_lstm_model.h5'.")
