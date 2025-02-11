import tkinter as tk
from tkinter import ttk
import librosa
import numpy as np
import simpleaudio as sa
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import ttk, filedialog, messagebox
from scipy.signal import butter, filtfilt

'''def on_row_double_click(event):
        """Handle double-click on a row."""
        selected_item = detection_tree.focus()  # Get the selected row's ID
        if selected_item:
            record = detection_tree.item(selected_item)["values"]  # Get the row's values
        show_recording(record[6],record[7],record[0],record[1],record[2],record[3])'''


def show_recording(path,file,time_start,time_end,freq_min,freq_max):
    
    path = path.split('output')[0]+'accoustic_data'
    file_path = path+'/'+file.split('duration')[0]+'.wav'
    if not file_path:
        return  # User canceled file dialog

    plot_frame = tk.Frame(spectogram_player)
    plot_frame.grid(column=0, row=1)

    try:
        y, sr = librosa.load(file_path)
        # Compute the mel spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

        # Convert to dB scale for better visualization
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Define the time and frequency intervals
        time_start = float(time_start)  # seconds
        time_end = float(time_end)    # seconds
        freq_min = float(freq_min)  # Hz
        freq_max = float(freq_max)  # Hz

        # Compute time and frequency indices
        time_indices = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr)
        freq_indices = librosa.mel_frequencies(n_mels=128, fmin=0, fmax=8000)

        # Mask the spectrogram to the desired intervals
        time_mask = (time_indices >= time_start) & (time_indices <= time_end)
        freq_mask = (freq_indices >= freq_min) & (freq_indices <= freq_max)

        S_filtered = S_dB[freq_mask][:, time_mask]

        # Plot the filtered spectrogram
        fig = plt.figure(figsize=(8, 3))
        ax = fig.add_subplot(111)
        img = librosa.display.specshow(
             S_filtered,
             sr=sr,
             x_axis="time",
             y_axis="mel",
             fmin=freq_min,
             fmax=freq_max,
             ax=ax,
             cmap="viridis",
             hop_length=librosa.time_to_samples(1 / sr))
               
        ax.set_title(f"Filtered Mel Spectrogram ({freq_min}-{freq_max} Hz, {time_start}-{time_end} s)")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")

        # Embed the plot in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas.draw()
        #canvas.get_tk_widget().pack(expand=True, fill="both")
    
    except Exception as e:
        messagebox.showerror("Error", f"Could not load file:\n{e}")

def bandpass_filter(data, lowcut, highcut, samplerate, order=5):
    """Apply a bandpass filter to the audio data."""
    nyquist = 0.5 * samplerate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def play_audio(audio_file,audio_data,sample_rate,start_time,end_time,low_freq,high_freq):
    if audio_file:
        try:
            # Handle stereo by taking one channel (left channel)
            if len(audio_data.shape) >1:
                audio_data = audio_data[:, 0]
            if low_freq ==0:
                low_freq = 1
            if high_freq==0:
                high_freq = 1

            # Extract the desired time interval
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            segment = audio_data[start_sample:end_sample]

            # Apply the bandpass filter
            filtered_segment = bandpass_filter(segment, low_freq, high_freq, sample_rate)
    
            # Normalize the data to fit in the range of int16 for playback
            filtered_segment = np.int16(filtered_segment / np.max(np.abs(filtered_segment)) * 32767)
    
            # Play the audio
            play_obj = sa.play_buffer(filtered_segment, 1, 2, sample_rate)
            play_obj.wait_done()

        except Exception as e:
            messagebox.showerror("Error", f"Could not play audio:\n{e}")
    else:
        messagebox.showwarning("No Audio", "Please load an audio file first.")

def show_recording(path,file,time_start,time_end,freq_min,freq_max):
    
    path = path.split('output')[0]+'accoustic_data'
    file_path = path+'/'+file.split('duration')[0]+'.wav'
    if not file_path:
        return  # User canceled file dialog

    plot_frame = tk.Frame(spectogram_player)
    plot_frame.grid(column=0, row=1)

    try:
        
        y, sr = librosa.load(file_path)
        # Compute the mel spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

        # Convert to dB scale for better visualization
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Define the time and frequency intervals
        time_start = float(time_start)  # seconds
        time_end = float(time_end)    # seconds
        freq_min = float(freq_min)  # Hz
        freq_max = float(freq_max)  # Hz

        # Compute time and frequency indices
        time_indices = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr)
        freq_indices = librosa.mel_frequencies(n_mels=128, fmin=0, fmax=8000)

        # Mask the spectrogram to the desired intervals
        time_mask = (time_indices >= time_start) & (time_indices <= time_end)
        freq_mask = (freq_indices >= freq_min) & (freq_indices <= freq_max)

        S_filtered = S_dB[freq_mask][:, time_mask]

        # Plot the filtered spectrogram
        #fig, ax = plt.figure(figsize=(10, 6))
        fig = plt.figure(figsize=(8, 3))
        ax = fig.add_subplot(111)
        img = librosa.display.specshow(
            S_filtered,
            sr=sr,
            x_axis="time",
            y_axis="mel",
            fmin=freq_min,
            fmax=freq_max,
            ax=ax,
            cmap="viridis",
            hop_length=librosa.time_to_samples(1 / sr))
               
        ax.set_title(f"Filtered Mel Spectrogram ({freq_min}-{freq_max} Hz, {time_start}-{time_end} s)")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")

        # Embed the plot in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas.draw()
        #canvas.get_tk_widget().pack(expand=True, fill="both")

        # Enable the play button
        play_button = ttk.Button(spectogram_player, text="Play Audio")
        play_button.place(relx=0.88, rely=0.2)

        # Enable the play button
        positive_button = ttk.Button(spectogram_player, text="Positive")
        positive_button.place(relx=0.88, rely=0.5)

        # Enable the play button
        negative_button = ttk.Button(spectogram_player, text="Negative")
        negative_button.place(relx=0.88, rely=0.6)
            
    except Exception as e:
        messagebox.showerror("Error", f"Could not load file:\n{e}")




# Create a new window (child window)
spectogram_player = tk.Tk()
spectogram_player.title("Wave File Spectrogram and Player")
spectogram_player.geometry("850x400")
spectogram_player.columnconfigure(0, weight=4)


# Enable the play button
play_button = ttk.Button(spectogram_player, text="Play Audio")
play_button.place(relx=0.88, rely=0.2)

# Enable the play button
positive_button = ttk.Button(spectogram_player, text="Positive")
positive_button.place(relx=0.88, rely=0.5)

# Enable the play button
negative_button = ttk.Button(spectogram_player, text="Negative")
negative_button.place(relx=0.88, rely=0.6)


spectogram_player.mainloop()