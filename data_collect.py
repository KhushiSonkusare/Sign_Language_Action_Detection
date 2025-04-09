import os
import cv2
import numpy as np
import mediapipe as mp
from gesture_collection import mediapipe_detection, draw_styled_landmarks, extract_keypoints

# Initialize Mediapipe Holistic model
mp_holistic = mp.solutions.holistic

# Define parameters
DATA_PATH = 'data2'  # Path to store the dataset
actions = ['please', 'good', 'game', 'help', 'you', 'play', 'guitar']  # New words for dataset
num_sequences = 30  # Number of sequences per word
sequence_length = 30  # Frames per sequence

# Function to create directory structure
def create_directory_structure():
    for action in actions:
        action_path = os.path.join(DATA_PATH, action)
        os.makedirs(action_path, exist_ok=True)
        for sequence in range(num_sequences):
            sequence_path = os.path.join(action_path, str(sequence))
            os.makedirs(sequence_path, exist_ok=True)

# Initialize video capture
cap = cv2.VideoCapture(0)  # Try different indices if needed

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Create directory structure
create_directory_structure()

try:
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in actions:
            for sequence in range(num_sequences):
                frame_count = 0
                print(f'Starting sequence {sequence} for action "{action}"')

                while frame_count < sequence_length:
                    ret, frame = cap.read()
                    if not ret:
                        print("Error: Failed to capture image.")
                        break
                    
                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)  # Draw landmarks for visual feedback

                    # Extract and save keypoints
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_count}.npy")
                    np.save(npy_path, keypoints)

                    frame_count += 1

                    # Display the frame (optional)
                    cv2.putText(image, f'Collecting frames for "{action}" - Sequence {sequence}, Frame {frame_count}', 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.imshow('Recording', image)

                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        print("Recording stopped by user.")
                        exit()
                
                print(f'Sequence {sequence} for action "{action}" recorded.')

            # Pause to confirm completion of action collection
            print(f'Finished all sequences for "{action}". Press any key to continue to the next action.')
            cv2.waitKey(0)  # Wait for any key press before moving to the next action

except Exception as e:
    print("An error occurred:", e)
finally:
    cap.release()
    cv2.destroyAllWindows()
