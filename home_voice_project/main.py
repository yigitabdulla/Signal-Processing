import os
import time
import librosa
import numpy as np
import matplotlib.pyplot as plt
import platform
import psutil
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM
from keras.utils import to_categorical
from keras.utils.np_utils import normalize


# Helper Function: Extract Features (MFCCs)
def extract_features(file_list):
    features, labels = [], []
    for file_path, label in file_list:
        try:
            audio, sr = librosa.load(file_path, sr=None)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)  # Mean of each MFCC
            features.append(mfcc_mean)
            labels.append(label)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    return np.array(features), np.array(labels)



# Helper Function: Print System Properties
def print_system_properties():
    print("\nSystem Properties:")
    print(f"Processor: {platform.processor()}")
    print(f"Machine: {platform.machine()}")
    print(f"Platform: {platform.platform()}")
    print(f"CPU Count: {psutil.cpu_count(logical=True)}")
    print(f"RAM: {round(psutil.virtual_memory().total / (1024 ** 3), 2)} GB\n")


# Helper Function: Plot Confusion Matrix
def plot_confusion_matrix(cm, classes, title="Confusion Matrix"):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, f"{cm[i, j]}", horizontalalignment="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()


# Load Dataset
def load_dataset(dataset_dir):
    class_names = []
    file_list = []
    file_counts = {}

    for class_name in os.listdir(dataset_dir):
        if class_name != "inohom":  # Exclude "inohom" class
            class_path = os.path.join(dataset_dir, class_name)
            files = os.listdir(class_path)
            file_list.extend([(os.path.join(class_path, file_name), class_name) for file_name in files])
            class_names.append(class_name)
            file_counts[class_name] = len(files)

    print("\nVoice File Counts:")
    for folder, count in file_counts.items():
        print(f"{folder}: {count}")

    return file_list, class_names


# Prepare Data
def prepare_data(file_list, class_names):
    labels = [class_names.index(label) for _, label in file_list]
    X_train, X_test, y_train, y_test = train_test_split(
        file_list, labels, test_size=0.2, stratify=labels, random_state=42
    )
    return X_train, X_test, y_train, y_test


# Machine Learning Section
def machine_learning_section(X_train, X_test, y_train, y_test, class_names):
    # Feature Extraction
    train_features, train_labels = extract_features(X_train)
    test_features, test_labels = extract_features(X_test)

    # Train Model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(train_features, train_labels)

    # Test Model and Measure Time
    start_time = time.time()
    predictions = clf.predict(test_features)
    end_time = time.time()

    # Metrics
    accuracy = accuracy_score(test_labels, predictions)
    print(f"\nMachine Learning Accuracy: {accuracy * 100:.2f}%")
    print(f"ML Test Duration: {end_time - start_time:.4f} seconds")

    # Plot Confusion Matrix
    cm = confusion_matrix(test_labels, predictions)
    plot_confusion_matrix(cm, class_names, title="ML Confusion Matrix")


# Deep Learning Section
def deep_learning_section(X_train, X_test, y_train, y_test, class_names):
    # Preprocess Data for Deep Learning Section
    def preprocess(file_list, labels):
        features = []
        for file_path, _ in file_list:
            try:
                audio, sr = librosa.load(file_path, sr=None)
                # Use the correct syntax with keyword arguments
                spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
                features.append(spectrogram)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
        return np.array(features), to_categorical(labels, num_classes=len(class_names))

    train_features, train_labels = preprocess(X_train, y_train)
    test_features, test_labels = preprocess(X_test, y_test)

    input_shape = train_features[0].shape

    # CNN Model
    cnn_model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(input_shape[0], input_shape[1], 1)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(len(class_names), activation="softmax")
    ])
    cnn_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    cnn_model.fit(train_features[..., np.newaxis], train_labels, epochs=10, batch_size=32, verbose=1)

    # Test CNN and Measure Time
    start_time = time.time()
    cnn_predictions = cnn_model.predict(test_features[..., np.newaxis])
    end_time = time.time()

    cnn_accuracy = accuracy_score(test_labels.argmax(axis=1), cnn_predictions.argmax(axis=1))
    print(f"\nCNN Accuracy: {cnn_accuracy * 100:.2f}%")
    print(f"CNN Test Duration: {end_time - start_time:.4f} seconds")

    cnn_cm = confusion_matrix(test_labels.argmax(axis=1), cnn_predictions.argmax(axis=1))
    plot_confusion_matrix(cnn_cm, class_names, title="CNN Confusion Matrix")

    # LSTM Model
    lstm_model = Sequential([
        LSTM(128, input_shape=(input_shape[1], input_shape[0])),
        Dense(128, activation="relu"),
        Dense(len(class_names), activation="softmax")
    ])
    lstm_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    lstm_model.fit(train_features.swapaxes(1, 2), train_labels, epochs=10, batch_size=32, verbose=1)

    # Test LSTM and Measure Time
    start_time = time.time()
    lstm_predictions = lstm_model.predict(test_features.swapaxes(1, 2))
    end_time = time.time()

    lstm_accuracy = accuracy_score(test_labels.argmax(axis=1), lstm_predictions.argmax(axis=1))
    print(f"\nLSTM Accuracy: {lstm_accuracy * 100:.2f}%")
    print(f"LSTM Test Duration: {end_time - start_time:.4f} seconds")

    lstm_cm = confusion_matrix(test_labels.argmax(axis=1), lstm_predictions.argmax(axis=1))
    plot_confusion_matrix(lstm_cm, class_names, title="LSTM Confusion Matrix")


# Helper Function: Plot Frequency Graphs
def plot_frequency_graph(file_path, title):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        plt.figure(figsize=(10, 4))
        plt.plot(audio)
        plt.title(title)
        plt.xlabel("Time (samples)")
        plt.ylabel("Amplitude")
        plt.show()
    except Exception as e:
        print(f"Error processing file {file_path} for plotting: {e}")

# Main Function
if __name__ == "__main__":
    dataset_dir = "sinyal isleme"
    file_list, class_names = load_dataset(dataset_dir)
    X_train, X_test, y_train, y_test = prepare_data(file_list, class_names)

    print_system_properties()

    print("\n--- Frequency Graph Section ---")
    for i, (file_path, label) in enumerate(file_list[:5]):  # Plot the first 5 files
        plot_frequency_graph(file_path, title=f"Frequency Graph: {label} - {os.path.basename(file_path)}")

    print("\n--- Machine Learning Section ---")
    machine_learning_section(X_train, X_test, y_train, y_test, class_names)

    print("\n--- Deep Learning Section ---")
    deep_learning_section(X_train, X_test, y_train, y_test, class_names)
