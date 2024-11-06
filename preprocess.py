import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the DSL-StrongPassword dataset
data = pd.read_csv('DSL-StrongPasswordData.csv')  # Adjust the file path if necessary

# Step 1: Calculate time differences
data['time_diff'] = data['H.t'].diff().fillna(0)  # Assuming 'H.t' contains timestamps of key presses

# Step 2: Create sequences of time differences
sequence_length = 10  # Define the length of the sequences
sequences = []
labels = []

# Group by 'subject' to create sequences for each user
for subject in data['subject'].unique():
    user_data = data[data['subject'] == subject]['time_diff'].values
    # Create sequences
    for i in range(len(user_data) - sequence_length):
        sequences.append(user_data[i:i + sequence_length])
        labels.append(1 if subject == 's033' else 0)  # Replace 's033' with the target subject

# Step 3: Convert sequences and labels to numpy arrays
X = np.array(sequences)
y = np.array(labels)

# Step 4: Normalize the sequences
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Reshape X to be 3D [samples, timesteps, features]
X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)  # 1 feature: time_diff

# Step 5: Save the preprocessed data for LSTM
# Saving X and y directly can help in loading later
np.savez('preprocessed_lstm_data.npz', X=X_scaled, y=y)

print("Preprocessing complete. Data saved as 'preprocessed_lstm_data.npz'.")
