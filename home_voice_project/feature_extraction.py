import os
import numpy as np
import pandas as pd
import librosa


# Function to read files from folders and extract features
def extract_features_from_folders(root_folder):
    features, labels = [], []

    # Walk through all subfolders in the root folder
    for label in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, label)

        if os.path.isdir(folder_path):  # Check if it's a directory
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".wav"):  # Only process .wav files
                    file_path = os.path.join(folder_path, file_name)
                    try:
                        audio, sr = librosa.load(file_path, sr=None)
                        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
                        mfcc_mean = np.mean(mfccs, axis=1)
                        features.append(mfcc_mean)
                        labels.append(label)
                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")

    return np.array(features), np.array(labels)


# Extract features
root_folder = "sinyal isleme"  # Your root folder path
features, labels = extract_features_from_folders(root_folder)

# Save to CSV
df = pd.DataFrame(features)
df['label'] = labels
df.to_csv('features.csv', index=False)

print("Features saved to features.csv")
