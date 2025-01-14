import os
import librosa
import numpy as np

##Bu kodu çalıştırmayın!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Path to the main folder
main_folder = "sinyal isleme deneme"

# List to store deleted files information
deleted_files = []

# Thresholds
short_duration_threshold = 2.0  # in seconds
silence_threshold = 0.01  # average amplitude threshold

# Function to check if the audio file is silent or has low volume
def is_low_volume(audio):
    avg_amplitude = np.mean(np.abs(audio))
    return avg_amplitude < silence_threshold

# Iterate through all subfolders and files
for folder_name in os.listdir(main_folder):
    folder_path = os.path.join(main_folder, folder_name)
    if os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".wav"):
                file_path = os.path.join(folder_path, file_name)
                try:
                    # Load the audio file
                    audio, sr = librosa.load(file_path)

                    # Check if the duration is too short
                    duration = librosa.get_duration(y=audio, sr=sr)
                    if duration < short_duration_threshold:
                        os.remove(file_path)
                        deleted_files.append((file_path, f"Deleted (Short duration: {duration:.2f} seconds)"))
                        continue

                    # Check if the file is silent or has low volume
                    if is_low_volume(audio):
                        os.remove(file_path)
                        deleted_files.append((file_path, "Deleted (Low volume)"))
                        continue

                except Exception as e:
                    deleted_files.append((file_path, f"Error: {str(e)}"))

# Save the deleted files list to a text file
with open("deleted_files_report.txt", "w") as report:
    for file_path, issue in deleted_files:
        report.write(f"{file_path} - {issue}\n")

print("Analysis complete. Deleted files report saved to 'deleted_files_report.txt'.")
