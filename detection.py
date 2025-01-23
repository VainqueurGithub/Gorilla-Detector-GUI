import tkinter as tk
import pandas as pd
import numpy as np
import os
import librosa
from scipy.io import wavfile
import simpleaudio as sa
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

def open_table_window():
    """Open a new window to populate and interact with the table."""
    def load_file():

        folder_path = filedialog.askdirectory(title="Select Folder")
        if folder_path:
            clear_table()
            # List files in the folder
            try:
                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)
                    with open(file_path, 'r') as fp:
                        lines = len(fp.readlines())
                    df = pd.read_csv(file_path, sep="\t")
                    conf_mean = round(df['conf'].mean(),2)
                    conf_min = round(df['conf'].min(),2)
                    conf_max = round(df['conf'].max(),2)
                    conf_std = round(np.std(df['conf'], ddof=1),2)  # ddof=1 for sample standard
                    file_duration = file_name.split("duration_")
                    file_duration = file_duration[1].split(".txt")
                    file_detection = lines-1
                    file_size = os.path.getsize(file_path)
                    file_duration = file_duration[0]
                    file_Path = folder_path
                    tree.insert("", "end", values=(file_name, file_detection, file_size, file_duration, conf_mean, conf_min, conf_max, conf_std,file_Path))
            except Exception as e:
                print(e)
    
    def clear_table():
        """Clear all rows from the Treeview table and reset selection."""
        # Remove all rows from the treeview
        for row in tree.get_children():
            tree.delete(row)
    
        tree.selection_remove(tree.selection())  # Remove any current selection            

    def on_record_select(event):
        """Display the selected record."""
        selected_item = tree.focus()  # Get the selected item (row ID)
        if selected_item:
            record = tree.item(selected_item)['values']  # Get the values of the selected row
            
            file_path = record[8]+'/'+record[0]

          
            if file_path:
                try:
                    for item in detection_tree.get_children():
                        detection_tree.delete(item)

                    df = pd.read_csv(file_path, sep="\t")
                    df['path'] = record[8]
                    df['file'] = record[0]
                    df = df[['Begin Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)', 'label', 'conf', 'path', 'file']]
                    df = df.round(2)
                    # Set up columns dynamically
                    detection_tree["columns"] = list(df.columns)
                    detection_tree["show"] = "headings"  # Hide default column

                    # Configure column headers
                    for col in df.columns:
                        detection_tree.heading(col, text=col)  # Set column heading
                        detection_tree.column(col, anchor="center", width=100)  # Adjust column width and alignment

                    # Populate table with DataFrame rows
                    for index, row in df.iterrows():
                        detection_tree.insert("", "end", values=list(row))
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load file: {e}")
    
    def show_recording(path,file,time_start,time_end,freq_min,freq_max):
        # Create a new window (child window)
        spectogram_player = tk.Toplevel(table_window)
        spectogram_player.title("Wave File Spectrogram and Player")
        spectogram_player.geometry("850x400")

        spectogram_player.columnconfigure(0, weight=4)
        #spectogram_player.columnconfigure(1, weight=1)
        
        audio_file = None
        audio_data = None
        sample_rate = None

        path = path.split('output')[0]+'accoustic_data'
        file_path = path+'/'+file.split('duration')[0]+'.wav'
        if not file_path:
            return  # User canceled file dialog

        plot_frame = tk.Frame(spectogram_player)
        #plot_frame.pack(expand=True, fill="both", padx=10, pady=10)
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
            play_button = ttk.Button(spectogram_player, text="Play Audio", command=play_audio(audio_file,audio_data,sample_rate))
            play_button.place(relx=0.88, rely=0.2)

            # Enable the play button
            positive_button = ttk.Button(spectogram_player, text="Positive")
            positive_button.place(relx=0.88, rely=0.5)

            # Enable the play button
            negative_button = ttk.Button(spectogram_player, text="Negative")
            negative_button.place(relx=0.88, rely=0.6)
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not load file:\n{e}")

    def play_audio(audio_file,audio_data,sample_rate):
        print("OHHHHHHHHH")
        """Play the loaded audio file."""
        '''if audio_file:
            try:
                # Convert audio data to 16-bit PCM format
                audio_data = (audio_data * (2**15 - 1) / np.max(np.abs(audio_data))).astype(np.int16)

                # Play audio using simpleaudio
                play_obj = sa.play_buffer(audio_data, 1, 2, sample_rate)  # Mono, 2 bytes per sample
                play_obj.wait_done()
            except Exception as e:
                messagebox.showerror("Error", f"Could not play audio:\n{e}")
        else:
            messagebox.showwarning("No Audio", "Please load an audio file first.")'''

    def on_row_double_click(event):
        """Handle double-click on a row."""
        selected_item = detection_tree.focus()  # Get the selected row's ID
        if selected_item:
            record = detection_tree.item(selected_item)["values"]  # Get the row's values
        show_recording(record[6],record[7],record[0],record[1],record[2],record[3])

    # Create the new window
    table_window = tk.Toplevel()
    table_window.title("Table Viewer")
    table_window.geometry("800x700")

    # Button to load the text file
    load_button = ttk.Button(table_window, text="Load output folder", command=load_file)
    load_button.pack(pady=10)

    # Treeview table to display files
    columns = ("File","Detections","Size (bytes)", "duration (s)", "conf_mean", "conf_min", "conf_max", "conf_std", "Path")
    tree = ttk.Treeview(table_window, columns=columns, show="headings", height=15)
    tree.heading("File", text="File")
    tree.heading("Detections", text="Events")
    tree.heading("Size (bytes)", text="Size (bytes)")
    tree.heading("duration (s)", text="Duration (s)")
    tree.heading("conf_mean", text="Score.Avg")
    tree.heading("conf_min", text="Score.Min")
    tree.heading("conf_max", text="Score.Max")
    tree.heading("conf_std", text="Score.Std")
    tree.heading("Path", text="Path")
    tree.column("File", anchor="e", width=300)
    tree.column("Detections", anchor="e", width=50)
    tree.column("Size (bytes)", anchor="e", width=70)
    tree.column("duration (s)", anchor="e", width=70)
    tree.column("conf_mean", anchor="e", width=60)
    tree.column("conf_min", anchor="e", width=60)
    tree.column("conf_max", anchor="e", width=60)
    tree.column("conf_std", anchor="e", width=50)
    tree.column("Path", anchor="e", width=30)
    tree.pack(pady=10, fill="both", expand=True)
    # Bind the treeview selection event
    tree.bind("<<TreeviewSelect>>", on_record_select)

    # Label to display the selected record

    detection_tree = ttk.Treeview(table_window)
    detection_tree.pack(pady=10, fill="both", expand=True)
    # Bind double-click event to the Treeview rows
    detection_tree.bind("<Double-1>", on_row_double_click)

