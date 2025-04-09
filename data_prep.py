import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Define constants
DATA_PATH = 'data2'  # Path to your data directory
ACTIONS = np.array(['please', 'good', 'game', 'help', 'you', 'play', 'guitar'])  # Updated actions
NO_SEQUENCES = 30  # Number of sequences per action
SEQUENCE_LENGTH = 30  # Frames per sequence

def load_data():
    sequences, labels = [], []
    label_map = {label: num for num, label in enumerate(ACTIONS)}  # Create label map
    for action in ACTIONS:
        for sequence in range(NO_SEQUENCES):
            window = []
            for frame_num in range(SEQUENCE_LENGTH):
                npy_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
                if os.path.exists(npy_path):  # Ensure file exists to avoid errors
                    res = np.load(npy_path)
                    window.append(res)
                else:
                    print(f"Warning: Missing file {npy_path}. Skipping...")
            sequences.append(window)
            labels.append(label_map[action])  # Use label map for labels

    X = np.array(sequences)  # Convert sequences to numpy array
    y = np.array(labels)     # Convert labels to numpy array
    y = to_categorical(y)    # One-hot encode labels

    return X, y

# Example usage
if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)
    print("Shape of X_train:", X_train.shape)
    print("Shape of X_test:", X_test.shape)
    print("Shape of y_train:", y_train.shape)
    print("Shape of y_test:", y_test.shape)
