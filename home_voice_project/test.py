import os
import time
import librosa
import numpy as np
import matplotlib.pyplot as plt
import platform
import psutil
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM
from keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

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

# Helper Function: Count Files in Each Folder
def count_files_in_folders(dataset_dir):
    folder_counts = {}
    for class_name in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_path):
            folder_counts[class_name] = len(os.listdir(class_path))
    return folder_counts

# Load Dataset
def load_dataset(dataset_dir):
    class_names = []
    file_list = []
    folder_counts = count_files_in_folders(dataset_dir)
    for class_name, count in folder_counts.items():
        print(f"{class_name}: {count} files")
    for class_name in folder_counts.keys():
        if class_name != "inohom":  # Exclude "inohom" class
            class_path = os.path.join(dataset_dir, class_name)
            for file_name in os.listdir(class_path):
                file_list.append((os.path.join(class_path, file_name), class_name))
            class_names.append(class_name)
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

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Support Vector Machine": SVC(kernel="linear", probability=True, random_state=42)
    }

    best_model = None
    best_accuracy = 0

    for model_name, model in models.items():
        # Train Model
        start_time = time.time()
        model.fit(train_features, train_labels)
        predictions = model.predict(test_features)
        end_time = time.time()

        # Metrics
        accuracy = accuracy_score(test_labels, predictions)
        print(f"\n{model_name} Accuracy: {accuracy * 100:.2f}%")
        print(f"{model_name} Test Duration: {end_time - start_time:.4f} seconds")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

        # Confusion Matrix
        cm = confusion_matrix(test_labels, predictions)
        plot_confusion_matrix(cm, class_names, title=f"{model_name} Confusion Matrix")

    print(f"\nBest Model: {best_model.__class__.__name__} with Accuracy: {best_accuracy * 100:.2f}%")


def deep_learning_section(X_train, X_test, y_train, y_test, class_names):
    # Preprocess Data for Deep Learning Section
    def preprocess(file_list, labels):
        features = []
        for file_path, _ in file_list:
            try:
                audio, sr = librosa.load(file_path, sr=None)
                spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
                features.append(spectrogram)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
        return np.array(features), to_categorical(labels, num_classes=len(class_names))

    train_features, train_labels = preprocess(X_train, y_train)
    test_features, test_labels = preprocess(X_test, y_test)

    input_shape = train_features[0].shape

    # Define Early Stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',   # Monitor validation loss
        patience=10,           # Stop after 3 epochs without improvement
        restore_best_weights=True  # Restore the best model weights
    )

    # CNN Model
    cnn_model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(input_shape[0], input_shape[1], 1)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(len(class_names), activation="softmax")
    ])
    cnn_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    history_cnn = cnn_model.fit(
        train_features[..., np.newaxis], train_labels,
        epochs=30, batch_size=32, verbose=1,
        validation_data=(test_features[..., np.newaxis], test_labels),
        callbacks=[early_stopping]  # Add EarlyStopping callback
    )

    # Test CNN and Measure Time
    start_time = time.time()
    cnn_predictions = cnn_model.predict(test_features[..., np.newaxis])
    end_time = time.time()

    cnn_accuracy = accuracy_score(test_labels.argmax(axis=1), cnn_predictions.argmax(axis=1))
    print(f"\nCNN Accuracy: {cnn_accuracy * 100:.2f}%")
    print(f"CNN Test Duration: {end_time - start_time:.4f} seconds")

    cnn_cm = confusion_matrix(test_labels.argmax(axis=1), cnn_predictions.argmax(axis=1))
    plot_confusion_matrix(cnn_cm, class_names, title="CNN Confusion Matrix")

    # Plot Training Loss Graph for CNN
    plt.figure(figsize=(10, 5))
    plt.plot(history_cnn.history['loss'], label='Training Loss')
    plt.plot(history_cnn.history['val_loss'], label='Validation Loss')
    plt.title('CNN Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # LSTM Model
    lstm_model = Sequential([
        LSTM(128, input_shape=(input_shape[1], input_shape[0])),
        Dense(128, activation="relu"),
        Dense(len(class_names), activation="softmax")
    ])
    lstm_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    history_lstm = lstm_model.fit(
        train_features.swapaxes(1, 2), train_labels,
        epochs=30, batch_size=32, verbose=1,
        validation_data=(test_features.swapaxes(1, 2), test_labels),
        callbacks=[early_stopping]  # Add EarlyStopping callback
    )

    # Test LSTM and Measure Time
    start_time = time.time()
    lstm_predictions = lstm_model.predict(test_features.swapaxes(1, 2))
    end_time = time.time()

    lstm_accuracy = accuracy_score(test_labels.argmax(axis=1), lstm_predictions.argmax(axis=1))
    print(f"\nLSTM Accuracy: {lstm_accuracy * 100:.2f}%")
    print(f"LSTM Test Duration: {end_time - start_time:.4f} seconds")

    lstm_cm = confusion_matrix(test_labels.argmax(axis=1), lstm_predictions.argmax(axis=1))
    plot_confusion_matrix(lstm_cm, class_names, title="LSTM Confusion Matrix")

    # Plot Training Loss Graph for LSTM
    plt.figure(figsize=(10, 5))
    plt.plot(history_lstm.history['loss'], label='Training Loss')
    plt.plot(history_lstm.history['val_loss'], label='Validation Loss')
    plt.title('LSTM Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

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
