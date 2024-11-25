# Import necessary libraries
import numpy as np
import os
import string
import mediapipe as mp
import cv2
from my_functions import *
import keyboard
# from tensorflow.keras.models import load_model
# NOTE : Workaround for Pycharm using miniconda python 3.10
from keras.api.models import load_model
import language_tool_python

# Set the path to the data directory
PATH = os.path.join('data')

# TODO : datasets : "Defines amount of different sets of alphabet data i.e from a different person or session".
datasets = os.listdir(PATH)

# Create an array of action labels by listing the contents of the data directory
actions = np.array(os.listdir(os.path.join(PATH, datasets[0])))

# Load the trained model
model = load_model('my_model.keras')

# Create an instance of the grammar correction tool
tool = language_tool_python.LanguageToolPublicAPI('en-UK')

# Initialize the lists
sentence, keypoints, last_prediction, grammar, grammar_result = [], [], [], [], []

# Flag to track landmark detection
landmark_detected = False

# Access the camera and check if the camera is opened successfully
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access camera.")
    exit()

# Create a holistic object for sign prediction
with mp.solutions.holistic.Holistic(min_detection_confidence=0.85, min_tracking_confidence=0.85) as holistic:
    # Run the loop while the camera is open
    while cap.isOpened():
        # Read a frame from the camera
        _, image = cap.read()
        
        # Make the image writable before calling draw_landmarks
        image.flags.writeable = True
        
        # Process the image and obtain sign landmarks using image_process function from my_functions.py
        results = image_process(image, holistic)
        
        # Check if any landmarks are detected
        landmark_detected = (
            results.pose_landmarks or 
            results.left_hand_landmarks or 
            results.right_hand_landmarks or 
            results.face_landmarks
        )
        
        # Only draw landmarks and process if landmarks are detected
        if landmark_detected:
            draw_landmarks(image, results)
            
            # Extract keypoints from the pose landmarks using keypoint_extraction function from my_functions.py
            keypoints.append(keypoint_extraction(results))          

            # Check if 10 frames have been accumulated
            if len(keypoints) == 10:
                keypoints = np.array(keypoints)
                prediction = model.predict(keypoints[np.newaxis, :, :])
                keypoints = []  # Clear for the next frames

                # Set a lower threshold (e.g., 0.7) to catch other letters
                if np.amax(prediction) > 0.75:
                    predicted_action = actions[np.argmax(prediction)]
                    if last_prediction != predicted_action:
                        sentence.append(predicted_action)
                        last_prediction = predicted_action

            # Limit the sentence length to 7 elements
            if len(sentence) > 7:
                sentence = sentence[-7:]

        # Reset if "Spacebar" is pressed
        if keyboard.is_pressed(' '):
            sentence, keypoints, last_prediction, grammar, grammar_result = [], [], [], [], []
            landmark_detected = False

        # Capitalize the first word of the sentence
        if sentence:
            sentence[0] = sentence[0].capitalize()

        # Combine last two elements if consecutive letters
        if len(sentence) >= 2:
            if sentence[-1] in string.ascii_letters:
                if sentence[-2] in string.ascii_letters or (sentence[-2] not in actions and sentence[-2] not in list(x.capitalize() for x in actions)):
                    sentence[-1] = sentence[-2] + sentence[-1]
                    sentence.pop(len(sentence) - 2)
                    sentence[-1] = sentence[-1].capitalize()

        # Perform grammar check if "Enter" is pressed
        if keyboard.is_pressed('enter'):
            text = ' '.join(sentence)
            grammar_result = tool.correct(text)

        # Display the sentence on the image only if landmarks are detected
        text_display = grammar_result if grammar_result else ' '.join(sentence)
        
        # Ensure the image is writable before displaying text
        image.flags.writeable = True
        
        # Calculate text position and draw the sentence only if landmarks are detected
        if landmark_detected:
            textsize = cv2.getTextSize(text_display, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_X_coord = (image.shape[1] - textsize[0]) // 2
            cv2.putText(image, text_display, (text_X_coord, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show the image on the display
        cv2.imshow('Camera', image)
        cv2.waitKey(1)

        # Check if the 'Camera' window was closed and break the loop
        if cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1:
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()
    tool.close()