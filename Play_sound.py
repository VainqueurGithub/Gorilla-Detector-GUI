import tkinter as tk
import subprocess

def run_script():
    try:
        # Replace 'your_script.py' with the path to your Python script
        subprocess.run(["python", "another_main_window,.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error while executing the script: {e}")

# Create the main window
root = tk.Tk()
root.title("Run Python Script")

# Add a button to run the script
run_button = tk.Button(root, text="Run Script", command=run_script)
run_button.pack(pady=20)

# Start the Tkinter event loop
root.mainloop()






 # Read the WAV file
            sample_rate,audio_data = wavfile.read(file_path)
            audio_file = file_path

            # Handle stereo by converting to mono
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)

            # Plot the spectrogram
            
            # Clear previous plot
            for widget in plot_frame.winfo_children():
                widget.destroy()

            # Create a new plot

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.specgram(audio_data, Fs=sample_rate, cmap="viridis")
            ax.set_title("Spectrogram")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
