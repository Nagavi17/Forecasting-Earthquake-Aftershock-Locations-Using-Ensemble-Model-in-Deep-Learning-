# -*- coding: utf-8 -*-
"""
Created on Sun May  5 12:55:27 2024

@author: Nagavi
"""
import numpy as np 
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout, SimpleRNN, GRU
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from tabulate import tabulate
from geopy.distance import geodesic

def load_dataset():
    df = pd.read_csv('earthquake_data.csv')
    X = df[['latitude', 'longitude', 'mag', 'depth']]
    # Label earthquakes with magnitude >= 3.5 as main events (1) and the rest as aftershocks (0)
    y = (df['mag'] >= 3.5).astype(int)
    return X, y

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(32, input_shape=input_shape),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_rnn_model(input_shape):
    model = Sequential([
        SimpleRNN(64, input_shape=input_shape),  # Increased units for RNN
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_gru_model(input_shape):
    model = Sequential([
        GRU(32, input_shape=input_shape),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    y_pred = (model.predict(X_test) > 0.5).astype("int32").reshape(-1)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = np.mean(y_pred == y_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return cm, accuracy, precision, recall, f1

def find_nearest_earthquake(aftershock_lat, aftershock_lon):
    min_distance = float('inf')
    nearest_earthquake = None
    for index, row in earthquake_data[earthquake_data['mag'] >= 6].iterrows():
        distance = geodesic((aftershock_lat, aftershock_lon), (row['latitude'], row['longitude'])).kilometers
        if distance < min_distance:
            min_distance = distance
            nearest_earthquake = row
    return min_distance, nearest_earthquake

# Set random seed for NumPy and TensorFlow
seed_value = 42
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

X, y = load_dataset()

# Perform the train-test split with a fixed random state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed_value)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape input data for LSTM, RNN, and GRU
input_shape = (1, X_train.shape[1])
X_train_reshaped = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test_reshaped = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Build LSTM model
lstm_model = build_lstm_model(input_shape)
# Train and evaluate LSTM model
lstm_cm, lstm_accuracy, lstm_precision, lstm_recall, lstm_f1 = train_and_evaluate_model(lstm_model, X_train_reshaped, y_train, X_test_reshaped, y_test)

# Build RNN model
rnn_model = build_rnn_model(input_shape)
# Train and evaluate RNN model
rnn_cm, rnn_accuracy, rnn_precision, rnn_recall, rnn_f1 = train_and_evaluate_model(rnn_model, X_train_reshaped, y_train, X_test_reshaped, y_test)

# Build GRU model
gru_model = build_gru_model(input_shape)
# Train and evaluate GRU model
gru_cm, gru_accuracy, gru_precision, gru_recall, gru_f1 = train_and_evaluate_model(gru_model, X_train_reshaped, y_train, X_test_reshaped, y_test)

# Print metrics table
metrics_data = [
    {"Model": "LSTM", "Accuracy": lstm_accuracy, "Precision": lstm_precision, "Recall": lstm_recall, "F1 Score": lstm_f1},
    {"Model": "RNN", "Accuracy": rnn_accuracy, "Precision": rnn_precision, "Recall": rnn_recall, "F1 Score": rnn_f1},
    {"Model": "GRU", "Accuracy": gru_accuracy, "Precision": gru_precision, "Recall": gru_recall, "F1 Score": gru_f1}
]

print(tabulate(metrics_data, headers="keys", tablefmt="grid"))


def plot_confusion_matrices(confusion_matrices, model_names):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, (cm, model_name) in enumerate(zip(confusion_matrices, model_names)):
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', ax=axes[i])
        axes[i].set_title(f'Confusion Matrix - {model_name}')
        axes[i].set_xlabel('Predicted labels')
        axes[i].set_ylabel('True labels')

    plt.tight_layout()
    plt.show()

# Gather confusion matrices and model names
confusion_matrices = [lstm_cm, rnn_cm, gru_cm]
model_names = ['LSTM', 'RNN', 'GRU']

# Plot all confusion matrices
plot_confusion_matrices(confusion_matrices, model_names)


# Read earthquake data from CSV file
earthquake_data = pd.read_csv('earthquake_data.csv')

# Create map
mymap = folium.Map(location=[np.mean(earthquake_data['latitude']), np.mean(earthquake_data['longitude'])], zoom_start=5)

# Add markers for earthquake locations
for index, row in earthquake_data.iterrows():
    color = 'red' if row['mag'] >= 6 else 'blue'
    folium.Marker(location=[row['latitude'], row['longitude']], popup=f'Magnitude: {row["mag"]}', icon=folium.Icon(color=color)).add_to(mymap)

# Add markers for aftershock locations near earthquakes
for index, row in earthquake_data[earthquake_data['mag'] < 6].iterrows():
    distance, nearest_earthquake = find_nearest_earthquake(row['latitude'], row['longitude'])
    if distance <= 10:  # Adjusted to 10 km radius
        folium.Marker(location=[row['latitude'], row['longitude']], popup=f'Aftershock: {row["mag"]}, Distance to earthquake: {distance:.2f} km', icon=folium.Icon(color='blue')).add_to(mymap)

# Save the map as an HTML file
mymap.save('earthquake_map_with_nearby_aftershocks.html')

# Display the map in default browser
import webbrowser
webbrowser.open('earthquake_map_with_nearby_aftershocks.html')
