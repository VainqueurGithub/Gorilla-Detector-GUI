import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sounddevice as sd
import tkinter as tk
import time

# Load audio file
audio_path = "C:/Detector GUI/model/datasets/accoustic_data/Screen Recording 2024-10-21 191021.wav"  # Replace with your file
y, sr = librosa.load(audio_path)

# Compute the spectrogram
hop_length = 512
n_fft = 2048
S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
S_dB = librosa.power_to_db(S, ref=np.max)

# Define the time range for the animation (in seconds)
start_time = 10  # Start time (in seconds)
end_time = 20    # End time (in seconds)

# Create a Tkinter window
root = tk.Tk()
root.title("Spectrogram with Moving Line")

# Create figure and plot spectrogram
fig, ax = plt.subplots()
img = librosa.display.specshow(S_dB, sr=sr, hop_length=hop_length, x_axis="time", y_axis="mel", ax=ax)
plt.colorbar(img, ax=ax, format="%+2.0f dB")
ax.set(title="Playing Spectrogram")

# Create a canvas to embed the plot in Tkinter
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

# Initialize the red vertical line at the start
line = ax.axvline(x=start_time, color="r", linestyle="--", lw=2)

# Define a function to update the line's position
def update_line():
    elapsed_time = sd.get_stream().time  # Get the time from sounddevice's stream
    if start_time <= elapsed_time <= end_time:
        line.set_xdata([elapsed_time])
        canvas.draw()
        root.after(10, update_line)  # Update every 10ms (adjust as needed)
    elif elapsed_time > end_time:
        # Ensure the line stops at the end time
        line.set_xdata([end_time])
        canvas.draw()

# Play the audio
sd.play(y, sr)

# Start the timer for the animation
root.after(10, update_line)  # Start updating the line
root.mainloop()
