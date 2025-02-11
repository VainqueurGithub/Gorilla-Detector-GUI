import tkinter as tk
import pandas as pd
import numpy as np
import os
#import simpleaudio as sa
import librosa
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
import threading

def open_table_window():
    """Open a new window to populate and interact with the table."""
    def load_file():

        folder_path = filedialog.askdirectory(title="Select Folder")
        if folder_path:
            clear_table()
            # List files in the folder
            try:
                for file_name in os.listdir(folder_path):
                    if file_name.endswith(".txt"):

                        file_path = os.path.join(folder_path, file_name)
                        with open(file_path, 'r') as fp:
                            lines = len(fp.readlines())
                        df = pd.read_csv(file_path, sep="\t")
                        conf_mean = round(df['conf'].mean(),2)
                        conf_min = round(df['conf'].min(),2)
                        conf_max = round(df['conf'].max(),2)
                        conf_std = round(np.std(df['conf'], ddof=1),2)  # ddof=1 for sample standard
                        file_duration = file_name.split("_duration_")
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
                    df = df[['Begin Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)', 'label', 'conf', 'path', 'file', 'chrunk']]
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
    
    def show_recording(path,file,time_start,time_end,freq_min,freq_max,chrunk,n_mels=128, hop_length=512):
        # Create a new window (child window)
        spectogram_player = tk.Toplevel(table_window)
        spectogram_player.title("Wave File Spectrogram and Player")
        spectogram_player.geometry("850x400")

        spectogram_player.columnconfigure(0, weight=4)
       
        path = path.split('output')[0]+'accoustic_data'
        file_path = path+'/'+file.split('_duration_')[0]+'.wav'
        if not file_path:
            return  # User canceled file dialog

        plot_frame = tk.Frame(spectogram_player)
        #plot_frame.pack(expand=True, fill="both", padx=10, pady=10)
        plot_frame.grid(column=0, row=1)

        try:
            global y, canvas, ax, sr  # Use global variables for Tkinter updates

            sr=None
            y, sr = librosa.load(file_path, sr=sr)
            # Compute the mel spectrogram
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

            # Convert to dB scale for better visualization
            S_dB = librosa.power_to_db(S, ref=np.max)

            start_time = 6*(int(chrunk)-1)
            end_time = 6*(int(chrunk))
            
            # Convert time to sample indices
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)

            # Extract segment of interest
            y_segment = y[start_sample:end_sample]

            # Create a Matplotlib Figure
            fig, ax = plt.subplots(figsize=(6, 3))
            canvas = FigureCanvasTkAgg(fig, master=spectogram_player)
            canvas.get_tk_widget().grid(row=3, column=0, columnspan=2)

            # Generate mel spectrogram
            spectrogram = librosa.feature.melspectrogram(y=y_segment, sr=sr, n_mels=n_mels, hop_length=hop_length)
            spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)  # Convert to dB scale

            # Clear previous plot
            ax.clear()

            # Plot spectrogram
            librosa.display.specshow(spectrogram_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel', ax=ax)
            ax.set_title(f"Spectrogram from {start_time:.2f}s to {end_time:.2f}s")
    
            # Update plot in Tkinter
            canvas.draw()

            # Plot the filtered spectrogram
            #fig, ax = plt.figure(figsize=(10, 6))
            '''fig = plt.figure(figsize=(8, 3))
            ax = fig.add_subplot(111)

            plt.figure(figsize=(10, 4))
            img = librosa.display.specshow(spectrogram_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.title("Spectrogram")
            plt.xlabel("Time (s)")
            plt.ylabel("Frequency (Mel)")
            fig.colorbar(img, ax=ax, format="%+2.0f dB")

            # Embed the plot in Tkinter
            canvas = FigureCanvasTkAgg(fig, master=plot_frame)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            canvas.draw()
            #canvas.get_tk_widget().pack(expand=True, fill="both")'''

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
    
    def bandpass_filter(data, lowcut, highcut, samplerate, order=5):
        """Apply a bandpass filter to the audio data."""
        nyquist = 0.5 * samplerate
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, data)
        return y

    def play_audio(path,file,start_time,end_time,low_freq,high_freq):
        
        path = path.split('output')[0]+'accoustic_data'
        file_path = path+'/'+file.split('_duration_')[0]+'.wav'
        '''if file_path:
            
            try:
                audio_data, sample_rate = librosa.load(file_path)
                # Handle stereo by taking one channel (left channel)
                start_time = float(start_time)
                end_time = float(end_time)
                low_freq = float(low_freq)
                high_freq = float(high_freq)

                if len(audio_data.shape) >1:
                    audio_data = audio_data[:, 0]
                if low_freq ==0:
                   low_freq = 1
                if high_freq ==0:
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
            messagebox.showwarning("No Audio", "Please load an audio file first.")'''
     
    def on_row_double_click(event):
        """Handle double-click on a row."""
        selected_item = detection_tree.focus()  # Get the selected row's ID
        if selected_item:
            record = detection_tree.item(selected_item)["values"]  # Get the row's values
        show_recording(record[6],record[7],record[0],record[1],record[2],record[3],record[8])
        '''try:
            # Start playback in a separate thread
            threading.Thread(target=play_audio, args=(record[6],record[7],record[0],record[1],record[2],record[3]), daemon=True).start()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to play audio: {e}")'''

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



    detection_tree.mainloop()