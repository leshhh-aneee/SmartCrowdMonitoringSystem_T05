import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
import pandas as pd
import joblib

class Predict:
    def __init__(self, wav_file, model_file, scaler_file, features_csv_file, update_interval=3, export_csv=False):
        self.wav_file = wav_file
        self.model_file = model_file
        self.scaler_file = scaler_file
        self.features_csv_file = features_csv_file
        self.update_interval = update_interval
        self.export_csv = export_csv

    def process_and_plot(self, start_time=None, end_time=None):
        try:
            sample_rate, data = wavfile.read(self.wav_file)
        except Exception as e:
            print(f"Error reading WAV file: {e}")
            return

        if data.ndim > 1:
            data = data[:, 0]

        max_abs_value = np.max(np.abs(data))
        if max_abs_value > 0:
            data = data / max_abs_value

        try:
            model = joblib.load(self.model_file)
            scaler = joblib.load(self.scaler_file)
        except Exception as e:
            print(f"Error loading model or scaler: {e}")
            return

        try:
            features_df = pd.read_csv(self.features_csv_file)
        except Exception as e:
            print(f"Error loading features CSV file: {e}")
            return

        if start_time is not None:
            start_idx = int(start_time * sample_rate)
            if start_idx < 0 or start_idx >= len(data):
                print("Invalid start time.")
                return
        else:
            start_idx = 0

        if end_time is not None:
            end_idx = int(end_time * sample_rate)
            if end_idx < 0 or end_idx >= len(data):
                print("Invalid end time.")
                return
        else:
            end_idx = len(data)

        BUFFER = min(1024 * 16, len(data))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        x_time = np.arange(BUFFER)
        x_freq = fftfreq(BUFFER, 1 / sample_rate)[:BUFFER // 2]

        line_time, = ax1.plot(x_time, data[:BUFFER], '-', lw=2)
        line_freq, = ax2.plot(x_freq, np.zeros(BUFFER // 2), '-', lw=2)

        ax1.set(title='AUDIO WAVEFORM', xlabel='Samples', ylabel='Amplitude', xlim=(0, BUFFER), ylim=(-1, 1))
        ax2.set(title='SPECTRUM', xlabel='Frequency (Hz)', ylabel='Magnitude', xlim=(0, sample_rate / 2))

        plt.subplots_adjust(hspace=0.4)
        plt.show(block=False)

        predictions = []
        time_stamps = []

        for start in range(start_idx, end_idx, BUFFER):
            end = min(start + BUFFER, end_idx)
            chunk = data[start:end]

            yf = fft(chunk)
            magnitude = np.abs(yf[:BUFFER // 2])

            if (start - start_idx) % (BUFFER * self.update_interval) == 0:
                line_time.set_ydata(chunk)
                line_freq.set_ydata(magnitude)
                ax2.set_ylim(0, max(np.max(magnitude), 1))
                fig.canvas.draw()
                fig.canvas.flush_events()

            row_idx = start // BUFFER
            if row_idx < len(features_df):
                # Adjust column selection to match the feature set used for training the model
                features = features_df.iloc[row_idx, :7].to_numpy().reshape(1, -1)  # Select correct number of features
                features_scaled = scaler.transform(features)  # Scale features
                prediction = model.predict(features_scaled)
                predictions.append(prediction[0])
                time_stamps.append((start + BUFFER) / sample_rate)

        if self.export_csv:
            df = pd.DataFrame({'Time (s)': time_stamps, 'Prediction': predictions})
            csv_file = self.wav_file.replace('.wav', '_predictions.csv')
            df.to_csv(csv_file, index=False)
            print(f'Predictions exported to {csv_file}')

        print('WAV file processing completed')
        print(f'Predictions: {predictions}')
