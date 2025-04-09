# 🤟 Sign Language Action Detection

This project implements a real-time sign language recognition system. It captures hand gestures via webcam, detects them using Mediapipe, interprets the action with a trained deep learning model, and forms coherent sentences from recognized gestures using the Groq API. Optional text-to-speech output is available for audio feedback.

---

## 🚀 Features

-   🎥 **Real-time Video Capture:** Uses OpenCV to capture video feed from the webcam.
-   🖐️ **Hand Gesture Detection:** Employs Mediapipe for accurate hand landmark detection.
-   🧠 **Action Classification:** Utilizes a TensorFlow/Keras LSTM-based deep learning model to classify detected gestures into sign language actions.
-   💬 **Sentence Coherence:** Integrates the Groq language model API to convert sequences of detected actions into grammatically correct and contextually meaningful sentences.
-   🔊 **Text-to-Speech (Optional):** Provides an option to convert the generated sentences into spoken audio using system TTS engines (e.g., via `pyttsx3`).
-   🔁 **Modular Workflow:** Includes separate Python scripts for:
    -   Data Collection (`data_collect.py`)
    -   Data Preprocessing (`data_prep.py`)
    -   Model Definition & Training (`model_def.py`)
    -   Real-time Inference & Application (`gesture_collection.py`)

---

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/KhushiSonkusare/Sign_Language_Action_Detection.git](https://github.com/KhushiSonkusare/Sign_Language_Action_Detection.git)
    cd Sign_Language_Action_Detection
    ```

2.  **Set up a virtual environment (Recommended):**
    ```bash
    # Create the environment
    python -m venv .venv

    # Activate the environment
    # On Windows:
    .\.venv\Scripts\activate
    # On macOS/Linux:
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    * The primary method is using the `requirements.txt` file:
        ```bash
        pip install -r requirements.txt
        ```
    * If `requirements.txt` is not available or missing dependencies, install manually:
        ```bash
        pip install opencv-python mediapipe tensorflow numpy groq pyttsx3 # Add any other required packages
        ```
        *(Note: Ensure `tensorflow` installation is compatible with your system/hardware. You might need `tensorflow-cpu` or specific GPU versions.)*

---

## 🧪 Usage

Follow these steps to run the different parts of the project:

1.  **Collect Gesture Data:**
    * Run the data collection script to record examples for each sign action you want to recognize.
        ```bash
        python data_collect.py
        ```

2.  **Preprocess the Collected Data:**
    * Prepare the collected landmark data into sequences suitable for training the LSTM model.
        ```bash
        python data_prep.py
        ```

3.  **Define and Train the Model:**
    * Train the LSTM model using the preprocessed data.
        ```bash
        python model_def.py
        ```

4.  **Run Real-Time Detection:**
    * Execute the main script to start the webcam feed, perform real-time gesture detection, action classification, sentence generation (via Groq), and optional TTS output.
        ```bash
        python gesture_collection.py
        ```

---

## 🧠 Model Info

-   **Model Type:** Long Short-Term Memory (LSTM) network, suitable for sequence-based data like gestures over time.
-   **Framework:** TensorFlow / Keras.
-   **Input:** Sequences of hand landmark coordinates extracted by Mediapipe over several frames.
-   **Output:** A probability distribution over the defined sign language actions/words. The action with the highest probability is chosen as the prediction.
-   **Training Data:** Trained on a custom dataset collected using the `data_collect.py` script.

---

## 🔗 Groq API Integration

-   **Purpose:** The Groq language model API is used to improve the fluency and coherence of the output. Instead of just listing detected signs (e.g., "hello", "how", "you"), it attempts to form a proper sentence (e.g., "Hello, how are you?").
-   **Setup:** You need to obtain an API key from [Groq](https://groq.com/). This key must be added to the `gesture_collection.py` script (or a configuration file, if used) where the Groq API is called. Look for a placeholder like `GROQ_API_KEY`.

---

## 📢 Text-to-Speech

-   **Functionality:** Optionally converts the final generated sentences into spoken audio.
-   **Implementation:** Typically uses libraries like `pyttsx3` (cross-platform) or system-specific TTS engines.
-   **Requirement:** Ensure the necessary TTS library (e.g., `pyttsx3`) is installed if you want to enable this feature.

---

## Author

**Khushi Sonkusare**

---