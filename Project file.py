import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Load the dataset
dataset = pd.read_csv('/mnt/data/Notes_Hours_Dataset.csv')

# Step 1: Data Preprocessing
# Convert 'time' to datetime format
dataset['time'] = pd.to_datetime(dataset['time'], format='%d-%m-%Y')

# Sort data by Instance and time
dataset.sort_values(by=['Instance', 'time'], inplace=True)

# Normalize the DG_Run_Hours column
scaler = MinMaxScaler()
dataset['DG_Run_Hours_Normalized'] = scaler.fit_transform(dataset[['DG_Run_Hours']])

# Encode the 'Instance' column
label_encoder = LabelEncoder()
dataset['Instance_Encoded'] = label_encoder.fit_transform(dataset['Instance'])

# Step 2: Prepare sequences for LSTM input
# Define a function to create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        seq_x = data[i:i + seq_length, :-1]  # Exclude target column for input
        seq_y = data[i + seq_length, -1]  # Include target column for output
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Group by Instance to create sequences per site
sequence_length = 30  # Example sequence length
X, y = [], []

for site in dataset['Instance_Encoded'].unique():
    site_data = dataset[dataset['Instance_Encoded'] == site][['DG_Run_Hours_Normalized', 'Instance_Encoded']].values
    site_X, site_y = create_sequences(site_data, sequence_length)
    X.append(site_X)
    y.append(site_y)

# Combine all site sequences
X = np.vstack(X)
y = np.concatenate(y)

# Reshape for LSTM input (samples, timesteps, features)
X = X.reshape(X.shape[0], sequence_length, X.shape[2])

# Step 3: Build LSTM Model
model = Sequential([
    Embedding(input_dim=len(label_encoder.classes_), output_dim=10, input_length=sequence_length),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1, activation='linear')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Step 4: Train the Model
history = model.fit(
    X, y,
    validation_split=0.2,
    epochs=20,
    batch_size=64,
    verbose=1
)

# Step 5: Evaluate and Save Model
loss, mae = model.evaluate(X, y, verbose=0)
print(f"Model Loss: {loss}, MAE: {mae}")

# Save the model
model.save('/mnt/data/lstm_dg_model.h5')
