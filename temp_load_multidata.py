import numpy as np
import os
from itertools import product
from keras.api.utils import to_categorical

# Set the path to the data directory
PATH = os.path.join('data')

# TODO : datasets : "Defines amount of different sets of alphabet data i.e from a different person or session".
datasets = os.listdir(PATH)
print(datasets)
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
                print(os.path.join(PATH, dataset_dir, action, str(sequence), str(frame) + '.npy'))
                temp.append(npy)

        landmarks.append(temp)
        labels.append(label_map[action])
    # Convert landmarks and labels to numpy arrays
    X, Y = np.array(landmarks), to_categorical(labels).astype(int)
else :
    print("Error : No dataset detected in \"data\" directory")
    exit(1)