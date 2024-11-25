# %%
# Import necessary libraries
import numpy as np
import os
from sklearn.model_selection import train_test_split
from itertools import product
from sklearn import metrics

from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.utils import to_categorical
# NOTE : Workaround for Pycharm using miniconda python 3.10
from keras.api.models import Sequential, load_model
from keras.api.utils import to_categorical




# Set the path to the data directory
PATH = os.path.join('data')

# TODO : datasets : "Defines amount of different sets of alphabet data i.e from a different person or session".
datasets = os.listdir(PATH)

# Exits if there's no dataset
if len(datasets) > 0:
    # Create an array of actions (signs) labels by listing the contents of the data directory
    actions = np.array(os.listdir(os.path.join(PATH, datasets[0])))

    # Define the number of sequences and frames
    sequences = 30
    frames = 10

    # Create a label map to map each action label to a numeric value
    label_map = {label: num for num, label in enumerate(actions)}

    # Initialize empty lists to store landmarks and labels
    landmarks, labels = [], []

    # Iterate over actions and sequences to load landmarks and corresponding labels
    # TODO : Modify to also load data from secondary data sources i.e 'data-1, data-2, ect' if it exists. Should result in more landmarks per label
    # TODO : This iterates over each subfolder under each label (alphabet).
    # TODO : Action is the label, sequence is the subfolder, frames is the individual .npy files.
    # TODO : Need to add a nest under sequence for each 'data-*' folder so looks up each dataset's 0th, 1st, ect sequence. Only changes the path.
    for action, sequence in product(actions, range(sequences)):
        temp = []
        for dataset_dir in datasets:
            for frame in range(frames):
                npy = np.load(os.path.join(PATH, dataset_dir, action, str(sequence), str(frame) + '.npy'))
                temp.append(npy)

        landmarks.append(temp)
        labels.append(label_map[action])
    # Convert landmarks and labels to numpy arrays
    X, Y = np.array(landmarks), to_categorical(labels).astype(int)

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=34, stratify=Y)

    # Define the model architecture
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(10, 126)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(Dropout(0.5))  # Added dropout for regularization
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))  # Another dropout layer
    model.add(Dense(actions.shape[0], activation='softmax'))

    # Compile the model with Adam optimizer and categorical cross-entropy loss
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    # Train the model with more epochs
    model.fit(X_train, Y_train, epochs=200, validation_split=0.1)  # Increased epochs and added validation split
    # Save the trained model
    model.save('my_model.keras')

    # Make predictions on the test set
    predictions = np.argmax(model.predict(X_test), axis=1)
    # Get the true labels from the test set
    test_labels = np.argmax(Y_test, axis=1)

    # Calculate the accuracy of the predictions
    accuracy = metrics.accuracy_score(test_labels, predictions)
else :
    print("Error : No dataset detected in \"data\" directory")
    exit(1)