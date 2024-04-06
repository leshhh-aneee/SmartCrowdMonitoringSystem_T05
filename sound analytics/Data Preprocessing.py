import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
import time
import pandas as pd

def process_and_plot_wav(wav_file, update_interval=3, export_csv=False, feature_selection_threshold=0.5):
    # Load the WAV file
    sample_rate, data = wavfile.read(wav_file)
    if data.ndim > 1:  # If stereo, take one channel
        data = data[:, 0]

    # Normalize the audio data
    data = data / np.max(np.abs(data))

    # Parameters for processing
    BUFFER = 1024 * 16  # samples per frame
    if len(data) < BUFFER:
        BUFFER = len(data)
    N = BUFFER

    # Number of buffers to average for 3-second intervals
    buffers_per_interval = int((3 * sample_rate) / BUFFER)
    freq_data_accumulator = np.zeros((buffers_per_interval, N // 2))
    interval_count = 0
    time_stamps = []

    # Frequency bins for FFT
    x_freq = fftfreq(N, 1 / sample_rate)[:N // 2]

    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    x_time = np.arange(0, BUFFER, 1)
    line_time, = ax1.plot(x_time, data[:BUFFER], '-', lw=2)
    line_freq, = ax2.plot(x_freq, np.zeros(N // 2), '-', lw=2)

    ax1.set_title('AUDIO WAVEFORM')
    ax1.set_xlabel('Samples')
    ax1.set_ylabel('Amplitude')
    ax1.set_xlim(0, BUFFER)
    ax1.set_ylim(-1, 1)

    ax2.set_title('SPECTRUM')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude')
    ax2.set_xlim(0, sample_rate / 2)

    plt.subplots_adjust(hspace=0.4)
    plt.show(block=False)

    start_time = time.time()
    all_fft_data = []

    for start_idx in range(0, len(data), BUFFER):
        end_idx = start_idx + BUFFER
        chunk = data[start_idx:end_idx]
        if len(chunk) < BUFFER:
            chunk = np.pad(chunk, (0, BUFFER - len(chunk)), 'constant')

        yf = fft(chunk)
        magnitude = np.abs(yf[:N // 2])
        line_time.set_ydata(chunk)
        line_freq.set_ydata(magnitude)
        ax2.set_ylim(0, np.max(magnitude))

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(update_interval)

        freq_data_accumulator[interval_count % buffers_per_interval, :] = magnitude
        interval_count += 1

        if interval_count % buffers_per_interval == 0:
            aggregated_magnitude = freq_data_accumulator.mean(axis=0)
            all_fft_data.append(aggregated_magnitude)
            time_stamp = (start_idx + BUFFER) / sample_rate
            time_stamps.append(time_stamp)
            freq_data_accumulator = np.zeros((buffers_per_interval, N // 2))  # Reset accumulator

    total_time = time.time() - start_time

    # Feature selection based on aggregated magnitude
    aggregated_magnitude = np.mean(all_fft_data, axis=0)
    significant_indices = aggregated_magnitude > np.max(aggregated_magnitude) * feature_selection_threshold
    significant_freqs = x_freq[significant_indices]
    significant_data = np.array(all_fft_data)[:, significant_indices]

    # Assuming bus presence is determined somehow for each interval
    bus_presence = [int(np.random.choice([0, 1])) for _ in time_stamps]  # Placeholder for actual bus presence logic

    print('WAV file processing completed')
    print(f'Total processing time: {total_time:.2f} seconds')
    print(f'Selected Frequencies: {significant_freqs}')

    if export_csv:
        columns = ['Time (s)'] + [f'Freq {freq:.2f} Hz' for freq in significant_freqs] + ['Bus Presence']
        data_to_export = np.column_stack((time_stamps, significant_data, bus_presence))
        df = pd.DataFrame(data_to_export, columns=columns)
        csv_file = wav_file.replace('.wav', '_svm_features_final_data.csv')
        df.to_csv(csv_file, index=False)
        print(f'Feature selected data exported to {csv_file}')

# Usage example
process_and_plot_wav('data_for_training.wav', export_csv=True, feature_selection_threshold=0.5)
