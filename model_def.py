import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

# Set your parameters
DATA_PATH = 'data2'  # Path to your data directory
ACTIONS = np.array(['please', 'good', 'game', 'help', 'you', 'play', 'guitar'])  # Updated actions
NO_SEQUENCES = 30  # Number of sequences per action
SEQUENCE_LENGTH = 30  # Frames per sequence

# Load data function
def load_data():
    X, y = [], []
    for idx, action in enumerate(ACTIONS):
        for sequence in range(NO_SEQUENCES):
            window = []
            for frame_num in range(SEQUENCE_LENGTH):
                npy_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
                res = np.load(npy_path)
                window.append(res)
            X.append(window)
            y.append(idx)
    
    return np.array(X), np.array(y)

# Load the dataset
X, y = load_data()

# One-hot encode the labels
y = tf.keras.utils.to_categorical(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# Create the model
def create_model(input_shape, num_classes):
    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint('models/best_model.keras', monitor='val_loss', save_best_only=True, mode='min')

    model = Sequential()
    model.add(LSTM(128, return_sequences=True, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))  # Increased dropout for better regularization

    model.add(LSTM(256, return_sequences=True, activation='relu'))  # Increased LSTM units
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(LSTM(128, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Lower learning rate
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    return model, tb_callback, early_stopping, checkpoint

# Example usage
input_shape = (30, 1662)  # Shape of your input data (should match the number of features per frame)
num_classes = len(ACTIONS)  # Number of classes
model, tb_callback, early_stopping, checkpoint = create_model(input_shape, num_classes)

# Fit the model
model.fit(X_train, y_train, epochs=200, validation_split=0.2, callbacks=[tb_callback, early_stopping, checkpoint])  # Train the model
model.summary()  # Print the model summary

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

# After training your model
model.save('models/action.keras')  # Save the model in the models directory
