import tkinter as tk
import pandas as pd
import numpy as np
import os
import librosa
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import sounddevice as sd
import time
import librosa.display

global spectogram_player
start_time = None

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
                messagebox.showerror("Error", f"Failed to load file: {e}")
    
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

                    # Check if column 'file' exists
                    if hasattr(df, 'file'):
                        df['path'] = record[8]
                    else:
                        df['path'] = record[8]
                        df['file'] = record[0]
                    df = df[['Begin Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)', 'label', 'conf', 'path', 'file', 'chrunk', 'decision']]
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
    
    def show_recording(path,file,time_start,time_end,freq_min,freq_max,chrunk,decision, n_mels=128, hop_length=512,n_fft = 2048):
        chrunk_po = chrunk
        # Create a new window (child window)
        spectogram_player = tk.Toplevel(table_window)
        spectogram_player.title("Wave File Spectrogram and Player")
        spectogram_player.geometry("850x400")

        spectogram_player.columnconfigure(0, weight=4)
        output_path = path
        path = path.split('output')[0]+'accoustic_data'
        file_path = path+'/'+file.split('_duration_')[0]+'.wav'
        if not file_path:
            return  # User canceled file dialog

        plot_frame = tk.Frame(spectogram_player)
        #plot_frame.pack(expand=True, fill="both", padx=10, pady=10)
        plot_frame.grid(column=0, row=1)

        try:
            global y, canvas, ax, sr  # Use global variables for Tkinter updates

            y, sr = librosa.load(file_path, sr=None)

            # Define the frame you want to analyze
            start_time = 6*(int(chrunk)-1)
            end_time = 6*(int(chrunk))

            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            frame = y[start_sample:end_sample]

            start_event = int(float(time_start) * sr)
            if start_event>=1:
                start_event-=1
            end_event = int(float(time_end) * sr)
            end_event+=1
            event = frame[start_event:end_event]
            # Compute the spectrogram
            
            # Compute STFT
            D = np.abs(librosa.stft(frame))

            # Convert to dB scale
            D_db = librosa.amplitude_to_db(D, ref=np.max)

            # Get frequency and time axes
            frequencies = librosa.fft_frequencies(sr=sr)
            times = librosa.frames_to_time(np.arange(D.shape[1]), sr=sr)

            # Define frequency range
            low_freq = 0  # Lower frequency limit (Hz)
            high_freq = 10500  # Upper frequency limit (Hz)

            # Create a Matplotlib Figure
            fig, ax = plt.subplots(figsize=(8, 3))
            canvas = FigureCanvasTkAgg(fig, master=spectogram_player)
            canvas.get_tk_widget().grid(row=3, column=0, columnspan=2)

             # Clear previous plot
            ax.clear()

            # Get frequency and time axes
            freq_indices = np.where((frequencies >= low_freq) & (frequencies <= high_freq))[0].astype(int)

            # Plot spectrogram for selected frequency range
            librosa.display.specshow(D_db[freq_indices, :], 
                         x_axis="time", 
                         y_axis="linear", 
                         sr=sr, 
                         cmap="magma", 
                         hop_length=512)

            plt.colorbar(label="Decibels (dB)")
            plt.title(f"Spectrogram ({low_freq}-{high_freq} Hz)")
            plt.xlabel("Time (s)")
            plt.ylabel("Frequency (Hz)")
            plt.ylim([low_freq, high_freq])
            plt.xlim([0, 6])



            #plt.show()
            
            if decision=='TP':
                # Draw the box
                if int(chrunk) > 1:
                    chrunk = int(chrunk)
                    plt.plot([float(time_start)-(6*(chrunk-1)), float(time_end)-(6*(chrunk-1)), float(time_end)-(6*(chrunk-1)), float(time_start)-(6*(chrunk-1)), float(time_start)-(6*(chrunk-1))],
                    [float(freq_min), float(freq_min), float(freq_max), float(freq_max), float(freq_min)],
                    color='green', linewidth=2)
                else:
                    plt.plot([float(time_start), float(time_end), float(time_end), float(time_start), float(time_start)],
                    [float(freq_min), float(freq_min), float(freq_max), float(freq_max), float(freq_min)],
                    color='green', linewidth=2)
            else:
                # Draw the box
                if int(chrunk) > 1:
                    chrunk = int(chrunk)
                    plt.plot([float(time_start)-(6*(chrunk-1)), float(time_end)-(6*(chrunk-1)), float(time_end)-(6*(chrunk-1)), float(time_start)-(6*(chrunk-1)), float(time_start)-(6*(chrunk-1))],
                    [float(freq_min), float(freq_min), float(freq_max), float(freq_max), float(freq_min)],
                    color='red', linewidth=2)
                else:
                    plt.plot([float(time_start), float(time_end), float(time_end), float(time_start), float(time_start)],
                    [float(freq_min), float(freq_min), float(freq_max), float(freq_max), float(freq_min)],
                    color='red', linewidth=2)

           # Embed Matplotlib figure in Tkinter
            canvas = FigureCanvasTkAgg(fig, master=spectogram_player)
            canvas_widget = canvas.get_tk_widget()
            canvas.get_tk_widget().grid(row=3, column=0, columnspan=2)
            
            # Initialize vertical line tracker
            line = ax.axvline(x=start_time, color="r", linestyle="--", lw=2)
            # Update plot in Tkinter
           

            def play_sound():
                #Define a function to update the line's position
                def update_line():
                    current_time = time.time() - start_time_global  # Get the elapsed time since play
                    if current_time < end_time:
                        line.set_xdata([current_time])
                        canvas.draw()
                        spectogram_player.after(10, update_line)  # Update every 10ms (adjust as needed)

                # Play the audio
                sd.play(frame, sr)

                # Start the timer for the animation
                start_time_global = time.time()

                # Begin the animation
                spectogram_player.after(10, update_line)  # Start updating the line

            def set_true_positive():

                output_path_file = output_path+'/'+file
                # Read the file content
                with open(output_path_file, 'r') as txt_file:
                    lines = txt_file.readlines()

                    # Define the column to update and the new value
                    column_to_update = 'decision'
                    new_value = 'TP'
                    

                    # Find the index of the column to update
                    header = lines[0].strip().split('\t')
                    column_index = header.index(column_to_update)

                    # Update the specific column value
                    for i in range(1, len(lines)):
                        row = lines[i].strip().split('\t')
                        if str(round(float(row[3]), 2)) == str(round(float(time_start), 2)) and str(round(float(row[4]), 2))==str(round(float(time_end), 2)) and str(round(float(row[5]), 2))==str(round(float(freq_min), 2)) and str(round(float(row[6]), 2))==str(round(float(freq_max), 2)) and str(round(float(row[9]), 2))==str(round(float(chrunk_po), 2)):
                            row[column_index] = new_value
                            lines[i] = '\t'.join(row) + '\n'

                    # Write the updated content back to the file
                    with open(output_path_file, 'w') as txt_file:
                        txt_file.writelines(lines)

            def set_false_positive():

                output_path_file = output_path+'/'+file
                # Read the file content
                with open(output_path_file, 'r') as txt_file:
                    lines = txt_file.readlines()

                    # Define the column to update and the new value
                    column_to_update = 'decision'
                    new_value = 'FP'
                    

                    # Find the index of the column to update
                    header = lines[0].strip().split('\t')
                    column_index = header.index(column_to_update)

                    # Update the specific column value
                    for i in range(1, len(lines)):
                        row = lines[i].strip().split('\t')
                        if str(round(float(row[3]), 2)) == str(round(float(time_start), 2)) and str(round(float(row[4]), 2))==str(round(float(time_end), 2)) and str(round(float(row[5]), 2))==str(round(float(freq_min), 2)) and str(round(float(row[6]), 2))==str(round(float(freq_max), 2)) and str(round(float(row[9]), 2))==str(round(float(chrunk_po), 2)):
                            row[column_index] = new_value
                            lines[i] = '\t'.join(row) + '\n'

                    # Write the updated content back to the file
                    with open(output_path_file, 'w') as txt_file:
                        txt_file.writelines(lines)

            # Play button
            play_button = ttk.Button(spectogram_player, text="Play Audio", command=play_sound)
            play_button.place(relx=0.88, rely=0.2)

            # Set TP button
            positive_button = ttk.Button(spectogram_player, text="Positive", command=set_true_positive)
            positive_button.place(relx=0.88, rely=0.5)

            # Set FP button
            negative_button = ttk.Button(spectogram_player, text="Negative", command=set_false_positive)
            negative_button.place(relx=0.88, rely=0.6)
  
        except Exception as e:
            messagebox.showerror("Error", f"Could not load file:\n{e}")
   
    def on_row_double_click(event):
        """Handle double-click on a row."""
        selected_item = detection_tree.focus()  # Get the selected row's ID
        if selected_item:
            record = detection_tree.item(selected_item)["values"]  # Get the row's values
        show_recording(record[6],record[7],record[0],record[1],record[2],record[3],record[8],record[9])
        
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