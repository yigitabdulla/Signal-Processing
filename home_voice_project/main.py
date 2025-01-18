import os
import pickle
import time
import librosa
import numpy as np
import matplotlib.pyplot as plt
import platform
import psutil
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, Dropout, BatchNormalization
from keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import load_model

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

# Helper Function: Plot Confusion Matrixex
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

def machine_learning_section(X_train, X_test, y_train, y_test, class_names):
    # Feature Extraction
    train_features, train_labels = extract_features(X_train)
    test_features, test_labels = extract_features(X_test)

    # Split training data for validation
    train_features_split, val_features, train_labels_split, val_labels = train_test_split(
        train_features, train_labels, test_size=0.2, random_state=42
    )

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
        model.fit(train_features_split, train_labels_split)
        predictions = model.predict(test_features)
        end_time = time.time()

        # Test Metrics
        accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions, average="weighted")
        recall = recall_score(test_labels, predictions, average="weighted")
        f1 = f1_score(test_labels, predictions, average="weighted")

        print(f"\n{model_name} Test Accuracy: {accuracy * 100:.2f}%")
        print(f"{model_name} Test Precision: {precision:.2f}")
        print(f"{model_name} Test Recall: {recall:.2f}")
        print(f"{model_name} Test F1-Score: {f1:.2f}")
        print(f"{model_name} Test Duration: {end_time - start_time:.4f} seconds")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

        # Confusion Matrix
        cm = confusion_matrix(test_labels, predictions)
        plot_confusion_matrix(cm, class_names, title=f"{model_name} Confusion Matrix")

    print(f"\nBest Model: {best_model.__class__.__name__} with Test Accuracy: {best_accuracy * 100:.2f}%")

    # Save the best model
    with open("best_model.pkl", "wb") as model_file:
        pickle.dump(best_model, model_file)

    # Test the best model
    test_start_time = time.time()
    final_predictions = best_model.predict(test_features)
    test_end_time = time.time()
    final_test_duration = test_end_time - test_start_time

    final_test_accuracy = accuracy_score(test_labels, final_predictions)
    final_precision = precision_score(test_labels, final_predictions, average="weighted")
    final_recall = recall_score(test_labels, final_predictions, average="weighted")
    final_f1 = f1_score(test_labels, final_predictions, average="weighted")

    print(f"\nBest Model Test Accuracy: {final_test_accuracy * 100:.2f}%")
    print(f"Best Model Test Precision: {final_precision:.2f}")
    print(f"Best Model Test Recall: {final_recall:.2f}")
    print(f"Best Model Test F1-Score: {final_f1:.2f}")
    print(f"Final Test Duration: {final_test_duration:.4f} seconds")

    # Save the final confusion matrix
    final_cm = confusion_matrix(test_labels, final_predictions)
    plot_confusion_matrix(final_cm, class_names, title="Best Model Confusion Matrix")


def deep_learning_section(X_train, X_test, y_train, y_test, class_names):
    # Preprocess Data for Deep Learning Section
    def preprocess(file_list, labels):
        features = []
        for file_path, _ in file_list:
            try:
                audio, sr = librosa.load(file_path, sr=None)
                spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
                spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
                spectrogram = (spectrogram - np.mean(spectrogram)) / np.std(spectrogram)
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
        patience=3,           # Stop after 3 epochs without improvement
        restore_best_weights=True  # Restore the best model weights
    )

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
        epochs=8, batch_size=32, verbose=1,
        validation_data=(test_features[..., np.newaxis], test_labels),
        callbacks=[early_stopping]  # Add EarlyStopping callback
    )

    # Test CNN and Measure Time
    start_time = time.time()
    cnn_predictions = cnn_model.predict(test_features[..., np.newaxis])
    end_time = time.time()

    cnn_accuracy = accuracy_score(test_labels.argmax(axis=1), cnn_predictions.argmax(axis=1))
    cnn_precision = precision_score(test_labels.argmax(axis=1), cnn_predictions.argmax(axis=1), average="weighted")
    cnn_recall = recall_score(test_labels.argmax(axis=1), cnn_predictions.argmax(axis=1), average="weighted")
    cnn_f1 = f1_score(test_labels.argmax(axis=1), cnn_predictions.argmax(axis=1), average="weighted")

    print(f"\nCNN Accuracy: {cnn_accuracy * 100:.2f}%")
    print(f"CNN Precision: {cnn_precision:.4f}")
    print(f"CNN Recall: {cnn_recall:.4f}")
    print(f"CNN F1-Score: {cnn_f1:.4f}")
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
        epochs=8, batch_size=32, verbose=1,
        validation_data=(test_features.swapaxes(1, 2), test_labels),
        callbacks=[early_stopping]  # Add EarlyStopping callback
    )

    # Save the trained LSTM model
    lstm_model.save("lstm_model.h5")
    print("LSTM model saved as 'lstm_model.h5'")

    # Test LSTM and Measure Time
    start_time = time.time()
    lstm_predictions = lstm_model.predict(test_features.swapaxes(1, 2))
    end_time = time.time()

    lstm_accuracy = accuracy_score(test_labels.argmax(axis=1), lstm_predictions.argmax(axis=1))
    lstm_precision = precision_score(test_labels.argmax(axis=1), lstm_predictions.argmax(axis=1), average="weighted")
    lstm_recall = recall_score(test_labels.argmax(axis=1), lstm_predictions.argmax(axis=1), average="weighted")
    lstm_f1 = f1_score(test_labels.argmax(axis=1), lstm_predictions.argmax(axis=1), average="weighted")

    print(f"\nLSTM Accuracy: {lstm_accuracy * 100:.2f}%")
    print(f"LSTM Precision: {lstm_precision:.4f}")
    print(f"LSTM Recall: {lstm_recall:.4f}")
    print(f"LSTM F1-Score: {lstm_f1:.4f}")
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

    print("#################################################### loaded model")
    # Load the saved LSTM model
    loaded_model = load_model("lstm_model.h5")
    print("Loaded LSTM model from 'lstm_model.h5'")

    # Make predictions with the loaded model
    loaded_model_predictions = loaded_model.predict(test_features.swapaxes(1, 2))
    loaded_model_accuracy = accuracy_score(test_labels.argmax(axis=1), loaded_model_predictions.argmax(axis=1))
    loaded_model_precision = precision_score(test_labels.argmax(axis=1), loaded_model_predictions.argmax(axis=1), average="weighted")
    loaded_model_recall = recall_score(test_labels.argmax(axis=1), loaded_model_predictions.argmax(axis=1), average="weighted")
    loaded_model_f1 = f1_score(test_labels.argmax(axis=1), loaded_model_predictions.argmax(axis=1), average="weighted")

    print(f"\nLoaded Model Accuracy: {loaded_model_accuracy * 100:.2f}%")
    print(f"Loaded Model Precision: {loaded_model_precision:.4f}")
    print(f"Loaded Model Recall: {loaded_model_recall:.4f}")
    print(f"Loaded Model F1-Score: {loaded_model_f1:.4f}")

    lstm_cm = confusion_matrix(test_labels.argmax(axis=1), loaded_model_predictions.argmax(axis=1))
    plot_confusion_matrix(lstm_cm, class_names, title="LSTM Confusion Matrix (Loaded Model)")

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

    folder_counts = {}

    # Count files in each folder
    for class_name in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_path):
            folder_counts[class_name] = len(os.listdir(class_path))

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.bar(folder_counts.keys(), folder_counts.values(), color='skyblue')
    plt.xlabel('Folder Name', fontsize=12)
    plt.ylabel('Number of Files', fontsize=12)
    plt.title('File Count in Each Folder', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    plt.show()

    gender_counts = {"Kadın": 0, "Erkek": 0}

    # Traverse the directory tree
    for root, _, files in os.walk(dataset_dir):
        for file_name in files:
            if "Kadın" in file_name:
                gender_counts["Kadın"] += 1
            elif "Erkek" in file_name:
                gender_counts["Erkek"] += 1

    # Plot the results
    plt.figure(figsize=(6, 4))
    plt.bar(gender_counts.keys(), gender_counts.values(), color=['lightcoral', 'lightskyblue'])
    plt.xlabel('Gender', fontsize=12)
    plt.ylabel('Number of Files', fontsize=12)
    plt.title('File Count by Gender Keywords', fontsize=14)
    plt.tight_layout()
    plt.show()

    sample = "sinyal isleme\kırmızı\kırmızı01_DUDU_Kadın_MVXVEU.wav"
    signal, sr = librosa.load(sample)

    #Waweshow
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(y=signal, sr=sr)
    plt.title("Wave")
    plt.show()

    #Spectrogram
    spec = np.abs(librosa.stft(signal))
    spec = librosa.amplitude_to_db(spec, ref=np.max)

    plt.figure(figsize=(14, 5))
    librosa.display.specshow(data=spec, sr=sr, x_axis="time", y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram")
    plt.show()

    #Mel-spectrogram
    mel_spect = librosa.feature.melspectrogram(y=signal, sr=sr)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

    plt.figure(figsize=(14, 5))
    librosa.display.specshow(mel_spect, y_axis="mel", x_axis="time")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel-spectrogram")
    plt.show()

    #Chromagram
    chroma = librosa.feature.chroma_cqt(y=signal, sr=sr, bins_per_octave=36)

    plt.figure(figsize=(14, 5))
    librosa.display.specshow(chroma, sr=sr, x_axis="time", y_axis="chroma", vmin=0, vmax=1)
    plt.colorbar()
    plt.show()

    #Mel-Frequency Cepstral Coefficients (MFCCs)
    mfccs = librosa.feature.mfcc(y=signal, sr=sr)

    plt.figure(figsize=(14, 5))
    librosa.display.specshow(mfccs, sr=sr, x_axis="time")
    plt.colorbar()
    plt.title("MFCCs")
    plt.show()

    print("\n--- Machine Learning Section ---")
    machine_learning_section(X_train, X_test, y_train, y_test, class_names)

    print("\n--- Deep Learning Section ---")
    deep_learning_section(X_train, X_test, y_train, y_test, class_names)

