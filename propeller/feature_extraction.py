import pandas as pd
import numpy as np
from scipy.signal import stft
import os


def calculate_all_file_features(file_path):
    # Load the CSV file
    data = pd.read_csv(file_path)

    # Select the columns for STFT transformation
    columns_to_transform = ['Speed', 'Voice', 'Acceleration X', 'Acceleration Y', 'Acceleration Z', 'Gyro X', 'Gyro Y',
                            'Gyro Z']

    # Dictionary to hold features
    stft_features = {}

    """
    Parameters for STFT
    nperseg: Specifies the number of overlapping points between consecutive segments.
    noverlap: Specifies the number of overlapping points between consecutive segments.
    High nperseg: Better frequency resolution, lower time resolution.
    Low nperseg: Better time resolution, lower frequency resolution.
    High noverlap: Smoother spectrogram, higher time resolution, higher computational cost.
    Low noverlap: Blockier spectrogram, lower time resolution, lower computational cost.
    """
    fs = 90  # Updated sample rate to 90 Hz
    nperseg = int(2 * fs)  # Common practice: twice the sampling frequency
    noverlap = int(nperseg * 0.75)  # 75% overlap is a common choice for balancing resolution

    # Function to calculate band features
    def calculate_band_features(frequencies, magnitude_spectrum, band):
        print(band)
        band_indices = np.where((frequencies >= band[0]) & (frequencies < band[1]))[0]
        if len(band_indices) == 0 or np.max(band_indices) >= magnitude_spectrum.shape[0]:
            return 0, 0, 0, 0, 0, 0, 0

        # Check if magnitude_spectrum has valid indices
        if magnitude_spectrum[band_indices, :].size == 0:
            return 0, 0, 0, 0, 0, 0, 0

        band_power = np.sum(magnitude_spectrum[band_indices, :])
        max_power = np.max(magnitude_spectrum[band_indices, :])
        min_power = np.min(magnitude_spectrum[band_indices, :])
        std_power = np.std(magnitude_spectrum[band_indices, :])
        median_power = np.median(magnitude_spectrum[band_indices, :])

        # Check if indices are valid for peak frequency calculation
        if len(band_indices) > 0 and np.argmax(magnitude_spectrum[band_indices, :]) < len(band_indices):
            peak_frequency = frequencies[band_indices[np.argmax(magnitude_spectrum[band_indices, :])]]
        else:
            peak_frequency = 0

        mean_frequency = np.sum(frequencies[band_indices] * magnitude_spectrum[band_indices, :].sum(axis=1)) / np.sum(
            magnitude_spectrum[band_indices, :])

        return band_power, max_power, min_power, std_power, median_power, peak_frequency, mean_frequency

    # Generate overlapping frequency bands with 10 Hz width and 5 Hz overlap
    bands = {chr(97 + i): (i * 5, i * 5 + 10) for i in range(18)}

    # Compute STFT and extract features for each column across the entire signal
    for col in columns_to_transform:
        if col in data.columns:
            # Perform STFT
            f, t, Zxx = stft(data[col].values, fs=fs, nperseg=nperseg, noverlap=noverlap)
            magnitude_spectrum = np.abs(Zxx)

            # Feature extraction across entire signal: band features
            for band_name, band_range in bands.items():
                band_power, max_power, min_power, std_power, median_power, peak_frequency, mean_frequency = calculate_band_features(
                    f, magnitude_spectrum, band_range)
                stft_features[f'{col}_{band_name}_power'] = [band_power]
                stft_features[f'{col}_{band_name}_max_power'] = [max_power]
                stft_features[f'{col}_{band_name}_min_power'] = [min_power]
                stft_features[f'{col}_{band_name}_std_power'] = [std_power]
                stft_features[f'{col}_{band_name}_median_power'] = [median_power]
                stft_features[f'{col}_{band_name}_peak_frequency'] = [peak_frequency]
                stft_features[f'{col}_{band_name}_mean_frequency'] = [mean_frequency]

    # Convert to a DataFrame with column names and a single row
    features_df = pd.DataFrame(stft_features)

    features_df.to_csv('output.csv', mode='a', index=False, header=not os.path.exists('output.csv'))


# Folder path
folder_path = './output'

# Walk through the folder and print all file paths
for root, dirs, files in os.walk(folder_path):
    for file in files:
        calculate_all_file_features(os.path.join(root, file))

print("Band power feature extraction completed and saved with column names.")
