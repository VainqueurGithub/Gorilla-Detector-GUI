import enum
import statistics
import tkinter as tk
from tkinter import ttk
from turtle import bgcolor
from unittest import TestResult
from PIL import Image, ImageTk
from tkinter import messagebox, filedialog
from tkinter import PhotoImage
import time
import threading
import random
import geopandas as gpd
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import maad
from maad import sound, features, rois
import librosa.display
import librosa
from ultralytics import YOLO
import cv2
from pydub import AudioSegment
from pydub.utils import make_chunks
import skimage.io
import torch
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg") # for backend
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from shapely.geometry import Point
from pyproj import CRS, Transformer
import seaborn as sns
sns.set_theme(style="darkgrid")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#save
# settings
hop_length = 512 # number of samples per time-step in spectrogram
n_mels = 128 # number of bins in spectrogram. Height of image
time_steps = 384 # number of time-steps. Width of image
spectrogram_width = 259  # Width of the spectrogram image
spectrogram_height = 128  # Height of the spectrogram image
audio_duration = 6.0  # Total audio duration in seconds
max_frequency = 5000.0  # Maximum frequency in Hz
Selection=1
View = 'Spectrogram 1'
Channel = 1
label='Gorilla'
i=0

# Count Input wav file
def count_wav_files(directory):
    # Ensure the directory exists
    if not os.path.isdir(directory):
        print(f"The directory {directory} does not exist.")
        return 0

    # Count .wav files
    wav_count = sum(1 for file in os.listdir(directory) if file.endswith((".wav", ".WAV")))
    return wav_count

# Count Input spectrogram file
def count_spect_files(directory):
    # Ensure the directory exists
    if not os.path.isdir(directory):
        print(f"The directory {directory} does not exist.")
        return 0

    # Count .wav files
    spect_count = sum(1 for file in os.listdir(directory) if file.endswith((".png", ".PNG")))
    return spect_count

# load audio. Using example from librosa
def select_folder_input():
    # Open a folder selection dialog
    input_path = filedialog.askdirectory(title="Select an input Folder")
    if input_path:  # Check if a folder was selected
        folder_input_path_label.config(text=f"Folder Selected: {input_path}")
        folder_input_path_label.selected_folder = input_path  # Store the selected folder
    else:
        folder_input_path_label.config(text="No folder selected")

def select_folder_output():
    # Open a folder selection dialog
    output_path = filedialog.askdirectory(title="Select an output Folder")
    if output_path:  # Check if a folder was selected
        folder_output_path_label.config(text=f"Folder Selected: {output_path}")
        folder_output_path_label.selected_folder = output_path  # Store the selected folder
    else:
        folder_output_path_label.config(text="No folder selected")

def proceed_action():
    # Check if a folder is selected
    if hasattr(folder_input_path_label, "selected_folder") and folder_input_path_label.selected_folder:
        if hasattr(folder_output_path_label, "selected_folder") and folder_output_path_label.selected_folder:
            return folder_output_path_label.selected_folder,folder_input_path_label.selected_folder
        else:
            messagebox.showerror("Error", "You must select an output folder before proceeding!")
    else:
        messagebox.showerror("Error", "You must select an input folder before proceeding!")


def clean_space_control():
    # Check if a folder is selected
    if (hasattr(folder_input_path_label, "selected_folder") and folder_input_path_label.selected_folder) and (hasattr(folder_output_path_label, "selected_folder") and folder_output_path_label.selected_folder):
        return folder_output_path_label.selected_folder,folder_input_path_label.selected_folder
    elif hasattr(folder_input_path_label, "selected_folder") and folder_input_path_label.selected_folder:
        return folder_input_path_label.selected_folder      
    elif hasattr(folder_output_path_label, "selected_folder") and folder_output_path_label.selected_folder:
        return folder_output_path_label.selected_folder
    else:
        messagebox.showerror("Error", "You must select an folder before proceeding!")

def predict_class(filename,conf, iou, show=False, imgsz=640,save=False,name='yolov8m'):
    model = YOLO('model/runs/detect/yolov8m_custom3/weights/best.pt')

    #Predicting
    result=model.predict(
        source=filename,show=show,imgsz=imgsz,save=save,name=name,conf=conf,iou=iou)
    return result

def audio_chunks(path_audio_chunks,input_path,filename):
    input_path = input_path+'/'+filename  # Replace with your file path
    audio = AudioSegment.from_file(input_path)
    
    # Define chunk duration in milliseconds (6 seconds = 6000 ms)
    chunk_duration_ms = 6000

    # Split the audio into chunks
    chunks = make_chunks(audio, chunk_duration_ms)

    # Create an output directory
    output_dir = path_audio_chunks
    os.makedirs(output_dir, exist_ok=True)
    
    chunks_name = []
    # Export each chunk as a separate audio file
    for i, chunk in enumerate(chunks):
        if len(chunk) < chunk_duration_ms:
            # Calculate the duration of silence needed
            silence_duration = chunk_duration_ms - len(chunk)
            silence = AudioSegment.silent(duration=silence_duration)
            
            # Append silence to the audio
            chunk = chunk + silence
        chunk_name = f"{output_dir}/{filename}_chunk_{i+1}.wav"  # Change extension if needed
        chunk.export(chunk_name, format="wav")  # Export as WAV format
        chunks_name.append(chunk_name)
        #return chunk_name
    return chunks_name

def scale_minmax(X, min=0.0, max=1.0):
    """Scale data to a specified range."""
    X_std = (X - X.min()) / (X.max() - X.min()) 
    X_scaled = X_std * (max - min) + min
    return X_scaled

def spectrogram_image(y, sr, out, hop_length, n_mels, fmax=3000):
    """Generate and save a spectrogram image."""
    # Generate log-mel spectrogram
    mels = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=hop_length*2, hop_length=hop_length, fmax=fmax
    )
    mels = np.log(mels + 1e-9)  # Add small value to avoid log(0)

    # Min-max scale to fit in 8-bit range
    img = scale_minmax(mels, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0)  # Flip so low frequencies are at the bottom
    img = 255 - img  # Invert colors: black==high energy
    # Save as PNG
    skimage.io.imsave(out, img)
    return img

def submit_form():
    # Retrieve data from input fields
    #input_folder = select_folder_input()
    output_path,input_path = proceed_action()
    path_audio_chunks = input_path+'/audio_chunks/'
    path_audio_chunk_spect = path_audio_chunks+'/chunk_spect/'
    proceed_action()
    treshold = treshold_entry.get()
    iou = iou_entry.get()
    
    # Basic validation
    if not treshold or not iou:
        messagebox.showerror("Error", "Treshold and IOU fields are required!")
        
    
    if treshold.isdigit():
        if float(treshold)>0 and float(treshold)<100:
            if iou.isdigit():
                if float(iou)>0 and float(iou)<100:
                    conf = float(treshold)/100
                    iou = float(iou)/100
                    start_task(conf,iou)
                else:
                    messagebox.showerror("Error", "IOU must be between 0 and 100!")
            else:
                messagebox.showerror("Error", "IOU must be a number!")
        else:
            messagebox.showerror("Error", "Treshold must be between 0 and 100!")
    else:     
        messagebox.showerror("Error", "Treshold must be a number!")
     
        
def start_task(conf, iou):
    
    # Disable the Start button
    run_button.config(state="disabled")
    # Open the popup window with the progress bar
    popup_progress_detect()
 
    # Run the task in a separate thread to avoid blocking the main GUI
    threading.Thread(target=task_progress_bar_detection(conf,iou)).start()


def popup_progress_detect():
    # Create a new popup window
    global popup, progress, label
    popup = tk.Toplevel(root)
    popup.title("Progress")
    popup.geometry("300x100")
    popup.resizable(False, False)

    # Add a label
    label = tk.Label(popup, text="Processing...")
    label.grid(column=1, row=10, sticky=tk.E, padx=5, pady=5)

    # Add the progress bar
    progress = ttk.Progressbar(popup, orient="horizontal", length=250, mode="determinate")
    progress.grid(column=1, row=11, sticky=tk.E, padx=5, pady=5)

    #threading.Thread(target=task_progress_bar_detection, args=(progress, label, popup, conf, iou), daemon=True).start()

def task_progress_bar_detection(conf,iou):
  
    output_path,input_path = proceed_action()
    path_audio_chunks = input_path+'/audio_chunks/'
    path_audio_chunk_spect = path_audio_chunks+'/chunk_spect/'
    directory_chunk = os.fsencode(path_audio_chunks)
    files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(".WAV") or f.endswith(".wav")]
    max_value = count_wav_files(input_path)
    progress["value"] = 0
    progress["maximum"] = max_value

    for i, file in enumerate(files, start=1):
        filename = os.path.basename(file)
        #computer file duration in second
        audio = AudioSegment.from_file(file)
        duration_ms = len(audio) # Get the duration in milliseconds
        duration_sec = duration_ms / 1000.0 # Convert milliseconds to seconds

        chunk_spectrogram(filename,input_path,path_audio_chunks,directory_chunk,conf,iou,output_path,duration_sec)
        time.sleep(0.3 + (0.3 * (i / max_value)))
        progress["value"] = i
        label.config(text=f"Processing {i}/{len(files)}")
        root.update_idletasks()  # Refresh the GUI

    # Close the popup when the task is done
    label.config(text="Detection completed!")
    popup.after(1000, popup.destroy)
    run_button.config(state="normal")


def detection_out_put(chunk_array,conf, iou, output_path,duration_sec):
    Selection=1
    View = 'Spectrogram 1'
    Channel = 1
    label='Gorilla'
    i=0
    detector_table = pd.DataFrame(columns=['Selection', 'View', 'Channel', 'Begin Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)', 'label', 'conf','hour_voc','date_voc','lat','long'])
    #filename_chunk_spect
    time_elaspe = 6
    for j,c in enumerate(chunk_array):
        if c.endswith((".png", ".PNG")):
            result = predict_class(c, conf, iou)
            for r in result:
                for cl in r.boxes:
                    bbox = cl.xywhn
                    x, y, width, height = bbox[0].tolist()  # Convert to individual scalar values
                    # Denormalize bounding box
                    x_min = (x - width / 2) * 259
                    x_max = (x + width / 2) * 259
                    y_min = (y - height / 2) * 128
                    y_max = (y + height / 2) * 128
                    hour_voc = random.randint(0, 23)
                    start_u = np.datetime64('2024-01-01').astype('M8[D]').astype('int64')
                    end_u = np.datetime64('2024-12-31').astype('M8[D]').astype('int64')
                    date_voc = np.datetime64(np.random.randint(start_u, end_u), 'D')
                    lat = random.uniform(514127, 566127)
                    long = random.uniform(9837988, 9859270)
                    # Convert bounding boxes coordinates to time and frequency ranges
                    if j==0:
                        time_start = (x_min / 259) * audio_duration
                        time_end = (x_max / 259) * audio_duration
                        freq_start = (y_min / 128) * max_frequency
                        freq_end = (y_max / 128) * max_frequency
                    else:
                        time_start = time_elaspe+((x_min / 259) * audio_duration)
                        time_end = time_elaspe+((x_max / 259) * audio_duration)
                        freq_start = (y_min / 128) * max_frequency
                        freq_end = (y_max / 128) * max_frequency
                        time_elaspe +=6
                    detector_table.loc[i] = [Selection, View, Channel,time_start,time_end,freq_start,freq_end,label,cl.conf[0].item(),hour_voc,date_voc,lat,long]
                    i+=1
                    Selection+=1

    filename_chunk_spect = os.path.basename(c)
    filename_chunk_spect = filename_chunk_spect.split(".wav")
    output_file = output_path+'/'+filename_chunk_spect[0]+'duration_'+str(duration_sec)+'.txt'
    detector_table.to_csv(output_file, sep="\t", index=False)


def chunk_spectrogram(filename,input_path,path_audio_chunks,directory_chunk,conf,iou,output_path,duration_sec,hop_length=hop_length, n_mels=n_mels, time_steps=time_steps):
    if filename.endswith((".wav", ".WAV")):
        time.sleep(0.1)  # Simulate processing time
        out_array = []
        #split de entire into 6 secs clips
        chunks_name = audio_chunks(path_audio_chunks,input_path,filename)
        for chunk_name in chunks_name:
            if os.path.basename(chunk_name).endswith((".wav", ".WAV")):
                y, sr = librosa.load(chunk_name, duration=10.0, sr=22050)
                out =path_audio_chunks+'/chunk_spect/'+os.path.basename(chunk_name)+'.png'
                out_array.append(out)
                # extract a fixed length window
                start_sample = 0 # starting at beginning
                length_samples = time_steps*hop_length
                window = y[start_sample:start_sample+length_samples]
                # Generate image and convert to PNG
                img = spectrogram_image(window, sr=sr, out=out, hop_length=hop_length, n_mels=n_mels)
                continue
            else:
                continue
        #Use the image (spectogram) for detection
        detection_out_put(out_array,conf, iou, output_path, duration_sec)

# CREATE MENU BUTTONS FUNCTIONS

# OUTPUT RESULT TEXT
def open_summary_result_window():
    # Create a new window (child window)
    summary_result_window = tk.Toplevel(root)
    summary_result_window.title("Detection results")
    summary_result_window.geometry("760x350")
    #login button

    # Treeview table to display files
    columns = ("File","Detections","Size (bytes)", "duration (s)", "conf_mean", "conf_min", "conf_max", "conf_std")
    file_table = ttk.Treeview(summary_result_window, columns=columns, show="headings", height=15)
    file_table.heading("File", text="File")
    file_table.heading("Detections", text="Events")
    file_table.heading("Size (bytes)", text="Size (bytes)")
    file_table.heading("duration (s)", text="Duration (s)")
    file_table.heading("conf_mean", text="Score.Avg")
    file_table.heading("conf_min", text="Score.Min")
    file_table.heading("conf_max", text="Score.Max")
    file_table.heading("conf_std", text="Score.Std")
    #file_table.heading("Path", text="Path")
    file_table.column("File", anchor="e", width=300)
    file_table.column("Detections", anchor="e", width=50)
    file_table.column("Size (bytes)", anchor="e", width=70)
    file_table.column("duration (s)", anchor="e", width=70)
    file_table.column("conf_mean", anchor="e", width=60)
    file_table.column("conf_min", anchor="e", width=60)
    file_table.column("conf_max", anchor="e", width=60)
    file_table.column("conf_std", anchor="e", width=50)
    #file_table.column("Path", anchor="e", width=100)
    file_table.pack(pady=10, fill="both", expand=True)

    # Scrollbar for the table
    scrollbar = ttk.Scrollbar(summary_result_window, orient="vertical", command=file_table.yview)
    file_table.configure(yscroll=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    return file_table

def marging_range_form():
    # Create a new window (child window)
    margin_form_window = tk.Toplevel(root)
    margin_form_window.title("Merging selection files")
    margin_form_window.geometry("400x400")
    margin_form_window.columnconfigure(0, weight=2)

    range1_label = ttk.Label(margin_form_window, text="Confidence Range 1 (%)")
    range1_label.grid(column=0, row=1, sticky=tk.W)

    range1_start_entry = ttk.Entry(margin_form_window, width=6)
    range1_start_entry.place(relx=0.37, rely=0)
    range1_start_entry.insert(0, 15)

    range1_label = ttk.Label(margin_form_window, text="to")
    range1_label.place(relx=0.50, rely=0)

    range1_end_entry = ttk.Entry(margin_form_window, width=6)
    range1_end_entry.place(relx=0.56, rely=0)
    range1_end_entry.insert(0, 25)

    range2_label = ttk.Label(margin_form_window, text="Confidence Range 2 (%)")
    range2_label.grid(column=0, row=2, sticky=tk.W)

    range2_start_entry = ttk.Entry(margin_form_window, width=6)
    range2_start_entry.place(relx=0.37, rely=0.05)
    range2_start_entry.insert(0, 26)

    range2_label = ttk.Label(margin_form_window, text="to")
    range2_label.place(relx=0.50, rely=0.05)

    range2_end_entry = ttk.Entry(margin_form_window, width=6)
    range2_end_entry.place(relx=0.56, rely=0.05)
    range2_end_entry.insert(0, 39)

    range3_label = ttk.Label(margin_form_window, text="Confidence Range 3 (%)")
    range3_label.grid(column=0, row=3, sticky=tk.W)

    range3_start_entry = ttk.Entry(margin_form_window, width=6)
    range3_start_entry.place(relx=0.37, rely=0.1)
    range3_start_entry.insert(0, 40)

    range3_label = ttk.Label(margin_form_window, text="to")
    range3_label.place(relx=0.50, rely=0.1)

    range3_end_entry = ttk.Entry(margin_form_window, width=6)
    range3_end_entry.place(relx=0.56, rely=0.1)
    range3_end_entry.insert(0, 100)

    #login button
    merge_button = ttk.Button(margin_form_window, text="Merge",width=15)
    merge_button.place(relx=0.7, rely=0.05)
    merge_button.bind("<Button>", lambda event: submit_mergin_form(event,range1_start_entry,range1_end_entry,range2_start_entry,range2_end_entry,range3_start_entry,range3_end_entry))

    # Separator object
    separator = ttk.Separator(margin_form_window, orient='horizontal')
    separator.place(relx=0, rely=0.2, relwidth=1, relheight=1)
    # Creating a photoimage object to use image 

    detection_map_button = ttk.Button(margin_form_window, text="Detection Map",command=map_detection_form)
    detection_map_button.place(relx=0.03, rely=0.24, relheight=0.1, relwidth=0.3)

    detection_plot_button = ttk.Button(margin_form_window, text="Detection Plots",command=visualization_form)
    detection_plot_button.place(relx=0.35, rely=0.24, relheight=0.1, relwidth=0.3)

def map_detection_form():
    detection_frame = pd.DataFrame(columns=['Selection', 'View', 'Channel', 'Begin Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)', 'label', 'conf','hour_voc', 'date_voc','lat','long'])
    gdf = pd.DataFrame(columns=['Selection', 'View', 'Channel', 'Begin Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)', 'label', 'conf','hour_voc', 'date_voc','lat','long'])
    folder_path = select_output_folder()

    try:
        for file_name in os.listdir(folder_path):
            data_frame = pd.read_csv(folder_path+'/'+file_name, sep='\t')
            detection_frame = pd.concat([detection_frame, data_frame], axis=0, ignore_index=True) # concatenating along rows
            geometry = [Point(xy) for xy in zip(round(detection_frame['lat']),round(detection_frame['long']))]
            gdf = gpd.GeoDataFrame(detection_frame, geometry=geometry) 
    except Exception as e:
        print(e) 

    current_directory = os.getcwd()
    shape_directory = current_directory+'/plots/map'
    
    for file in os.listdir(shape_directory):
        # check the files which are end with specific extension
	    if file.endswith(".shp"):
              data = gpd.read_file(os.path.join(shape_directory, file))
              if gdf.empty:
                  data.plot(color="lightgrey", edgecolor="black", alpha=0.7)
                  plt.title("Detection distribution Map.")
                  plt.show()
              gdf.plot(ax = data.plot(color="lightgrey", edgecolor="black", alpha=0.7), marker='o', color='red', markersize=10)
              plt.title("Detection distribution Map.")
              plt.show()
              

def visualization_form():
    global middleFrame
    detection_frame = pd.DataFrame(columns=['Selection', 'View', 'Channel', 'Begin Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)', 'label', 'conf','hour_voc', 'date_voc','lat','long'])
    folder_path = select_output_folder()

    try:
        for file_name in os.listdir(folder_path):
            data_frame = pd.read_csv(folder_path+'/'+file_name, sep='\t')
            detection_frame = pd.concat([detection_frame, data_frame], axis=0, ignore_index=True) # concatenating along rows
    except Exception as e:
        print(e) 
    
    detection_frame['date_voc'] = pd.to_datetime(detection_frame.date_voc)
    detection_frame['Month'] = detection_frame['date_voc'].dt.month
    detection_frame['Year'] = detection_frame['date_voc'].dt.year
    detection_frame.set_index(pd.DatetimeIndex(detection_frame['date_voc']), inplace=True)

    fig, axes = plt.subplots(2,2, figsize=(17,5))
    lineplot = sns.lineplot(x="hour_voc", y="conf",hue="Month",data=detection_frame, ax=axes[0,0])
    lineplot.axes.set_title('Detection confidence over time')
    lineplot = sns.lineplot(x="hour_voc", y="High Freq (Hz)",hue="Month",data=detection_frame, ax=axes[0,1])
    lineplot.axes.set_title('Frequency Distribution over time')


    lineplot = sns.histplot(data=detection_frame, x="hour_voc", ax=axes[1,0])
    lineplot.axes.set_title('Hourly detection')
    lineplot = sns.histplot(data=detection_frame, x="Month", ax=axes[1,1])
    lineplot.axes.set_title('Monthy Detection')
    plt.show()
def submit_mergin_form(event,range1_start_entry,range1_end_entry,range2_start_entry,range2_end_entry,range3_start_entry,range3_end_entry):
    range1_start_entry = range1_start_entry.get()
    range1_end_entry = range1_end_entry.get()
    range2_start_entry = range2_start_entry.get()
    range2_end_entry = range2_end_entry.get()
    range3_start_entry = range3_start_entry.get()
    range3_end_entry = range3_end_entry.get()
    if not range1_start_entry or not range1_end_entry or not range2_start_entry or not range2_end_entry or not range3_start_entry or not range3_end_entry:
        messagebox.showerror("Error", "All range are required. Please fill by zero")
        
    if range1_start_entry.isdigit() and range1_end_entry.isdigit():
        if float(range1_start_entry)>=0 and float(range1_end_entry)<=100 and float(range1_end_entry)>=float(range1_start_entry):
            if range2_start_entry.isdigit() and range2_start_entry.isdigit():
                if float(range2_start_entry)>=0 and float(range2_end_entry)<=100 and float(range2_end_entry)>=float(range2_start_entry):
                    if range3_start_entry.isdigit() and range3_end_entry.isdigit():
                        if float(range3_start_entry)>=0 and float(range3_end_entry)<=100 and float(range3_end_entry)>=float(range3_start_entry):
                           margin_selection(range1_start_entry,range1_end_entry,range2_start_entry,range2_end_entry,range3_start_entry,range3_end_entry)
                        else:
                            messagebox.showerror("Error", "Something is wrong within range confidence 3")
                    else:
                        messagebox.showerror("Error", "Something is wrong within range confidence 3")
                else:
                    messagebox.showerror("Error", "Something is wrong within range confidence 2")
            else:
                messagebox.showerror("Error", "Something is wrong within range confidence 2")
        else:
            messagebox.showerror("Error", "Something is wrong within range confidence 1")
    else:     
        messagebox.showerror("Error", "Something is wrong within range confidence 1")

def margin_selection(range1_start_entry,range1_end_entry,range2_start_entry,range2_end_entry,range3_start_entry,range3_end_entry):
    detection_range1 = pd.DataFrame(columns=['Selection', 'View', 'Channel', 'Begin Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)', 'label', 'conf','hour_voc', 'date_voc','lat','long'])
    detection_range2 = pd.DataFrame(columns=['Selection', 'View', 'Channel', 'Begin Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)', 'label', 'conf','hour_voc', 'date_voc','lat','long'])
    detection_range3 = pd.DataFrame(columns=['Selection', 'View', 'Channel', 'Begin Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)', 'label', 'conf','hour_voc', 'date_voc','lat','long'])
    folder_path = select_output_folder()

    range1_start = float(range1_start_entry)/100
    range1_end = float(range1_end_entry)/100
    range2_start = float(range2_start_entry)/100
    range2_end = float(range2_end_entry)/100
    range3_start = float(range3_start_entry)/100
    range3_end = float(range3_end_entry)/100

    try:
        for file_name in os.listdir(folder_path):
            data = pd.read_csv(folder_path+'/'+file_name, sep='\t')
            data_range1 = data[(data['conf']>range1_start) & (data['conf']<range1_end)]
            data_range2 = data[(data['conf']>range2_start) & (data['conf']<range2_end)]
            data_range3 = data[(data['conf']>range3_start) & (data['conf']<range3_end)]
            detection_range1 = pd.concat([detection_range1, data_range1], axis=0, ignore_index=True) # concatenating along rows
            detection_range2 = pd.concat([detection_range2, data_range2], axis=0, ignore_index=True) # concatenating along rows
            detection_range3 = pd.concat([detection_range3, data_range3], axis=0, ignore_index=True) # concatenating along rows

        detection_range1.to_csv(folder_path+'/range1.txt', sep="\t", index=False)
        detection_range2.to_csv(folder_path+'/range2.txt', sep="\t", index=False)
        detection_range3.to_csv(folder_path+'/range3.txt', sep="\t", index=False)
    except Exception as e:
        print(e)
        
def select_output_folder():
    """Open a dialog to select a folder and list its files."""
    folder_path = filedialog.askdirectory(title="Select Folder")
    if folder_path:
        return folder_path
def populate_table():
    folder_path = select_output_folder()
    populate_summary_result_table(folder_path)

def populate_summary_result_table(folder_path):
    """Populate the Treeview table with files from the selected folder."""
    # Clear existing data
    file_table = open_summary_result_window()
    for item in file_table.get_children():
        file_table.delete(item)
    
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
            #file_Path = folder_path
            file_table.insert("", "end", values=(file_name, file_detection, file_size, file_duration, conf_mean, conf_min, conf_max, conf_std))
    except Exception as e:
        print(e)

def clean_space():
    """Display a warning message when the button is clicked."""
    response = messagebox.askquestion("Warning", "Do you want to delete the input data and its results ?")

    if response == 'yes':
        output_path,input_path = clean_space_control()
        path_audio_chunks = input_path+'/audio_chunks/'
        path_audio_chunk_spect = path_audio_chunks+'/chunk_spect/'
        
        """Delete all files in the selected folder."""
    if not output_path and not input_path:
        messagebox.showwarning("No Folder Selected", "Please select either input folder or output folder.")
        return

    try:
        if output_path and input_path:
            file_count = 0
            for filename in os.listdir(output_path):
                file_path = os.path.join(output_path, filename)
                if os.path.isfile(file_path):  # Only delete files, not subfolders
                    os.remove(file_path)
                    file_count += 1
            for filename in os.listdir(input_path):
                file_path = os.path.join(input_path, filename)
                if os.path.isfile(file_path):  # Only delete files, not subfolders
                    os.remove(file_path)
                    file_count += 1

            for filename in os.listdir(path_audio_chunks):
                file_path = os.path.join(path_audio_chunks, filename)
                if os.path.isfile(file_path):  # Only delete files, not subfolders
                    os.remove(file_path)
                    file_count += 1

            for filename in os.listdir(path_audio_chunk_spect):
                file_path = os.path.join(path_audio_chunk_spect, filename)
                if os.path.isfile(file_path):  # Only delete files, not subfolders
                    os.remove(file_path)
                    file_count += 1
            messagebox.showinfo("Success", f"Deleted {file_count} files from the folder.")
        elif input_path:
            file_count = 0
            for filename in os.listdir(input_path):
                file_path = os.path.join(input_path, filename)
                if os.path.isfile(file_path):  # Only delete files, not subfolders
                    os.remove(file_path)
                    file_count += 1
            for filename in os.listdir(path_audio_chunks):
                file_path = os.path.join(path_audio_chunks, filename)
                if os.path.isfile(file_path):  # Only delete files, not subfolders
                    os.remove(file_path)
                    file_count += 1

            for filename in os.listdir(path_audio_chunk_spect):
                file_path = os.path.join(path_audio_chunk_spect, filename)
                if os.path.isfile(file_path):  # Only delete files, not subfolders
                    os.remove(file_path)
                    file_count += 1
            messagebox.showinfo("Success", f"Deleted {file_count} files from the folder.")

        elif output_path:
            file_count = 0
            for filename in os.listdir(output_path):
                file_path = os.path.join(output_path, filename)
                if os.path.isfile(file_path):  # Only delete files, not subfolders
                    os.remove(file_path)
                    file_count += 1
            messagebox.showinfo("Success", f"Deleted {file_count} files from the folder.")
           
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# root window
root = tk.Tk()
root.geometry("800x520")
root.title('Gorilla Detector')
root.resizable(0, 0)

# configure the grid
root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=3)
root.columnconfigure(2, weight=1)



# Load the image as a PhotoImage object
icon_image = PhotoImage(file="images/icon.png")  # Replace with the path to your .png file
# Set the window icon
root.wm_iconphoto(True, icon_image)
# Window size


save_model_output_var = tk.IntVar()
show_detection_var = tk.IntVar()


mergin_button = ttk.Button(root, text="Merging and Ploting",width=20, command=marging_range_form)
mergin_button.grid(column=0, row=1, sticky=tk.N, padx=5, pady=5)
open_button = ttk.Button(root, text="Detections", command=populate_table)
open_button.grid(column=1, row=1, sticky=tk.W, padx=5, pady=5)
age_sexe_button = ttk.Button(root, text="Estimate Age/Sex")
age_sexe_button.grid(column=1, row=1, sticky=tk.N, padx=5, pady=5)
call_type_button = ttk.Button(root, text="Estimate Call type")
call_type_button.grid(column=1, row=1, sticky=tk.E, padx=5, pady=5)


# Separator object
separator = ttk.Separator(root, orient='horizontal')
separator.place(relx=0, rely=0.1, relwidth=1, relheight=1)


# Form

# Add a button to trigger the folder selection dialog
select_input_folder_button = ttk.Button(root, text="Select Input Folder", command=select_folder_input, width=20)
select_input_folder_button.place(relx=0.03, rely=0.2)
#grid(column=0, row=4, sticky=tk.W, padx=5, pady=5)

# Add a label to display the selected folder path
folder_input_path_label = ttk.Label(root, text="No folder selected", wraplength=1200)
folder_input_path_label.place(relx=0.25, rely=0.2)

# Add a button to trigger the folder selection dialog
select_output_folder_button = ttk.Button(root, text="Select Output Folder", command=select_folder_output, width=20)
select_output_folder_button.place(relx=0.03, rely=0.3)

# Add a label to display the selected folder path
folder_output_path_label = ttk.Label(root, text="No folder selected", wraplength=1200)
folder_output_path_label.place(relx=0.25, rely=0.3)

# Add a checkbox
save_model_output_label = ttk.Label(root, text="Save model output")
save_model_output_label.place(relx=0.06, rely=0.4)
checkbox1 = ttk.Checkbutton(root, variable=save_model_output_var)
checkbox1.place(relx=0.03, rely=0.4)

show_detection_label = ttk.Label(root, text="Show detection")
show_detection_label.place(relx=0.23, rely=0.4)
checkbox2 = ttk.Checkbutton(root, variable=show_detection_var)
checkbox2.place(relx=0.2, rely=0.4)


# Label to display the status

treshold_label = ttk.Label(root, text="Treshold (%):")
treshold_label.place(relx=0.55, rely=0.4)

treshold_entry = ttk.Entry(root, width=6)
treshold_entry.place(relx=0.65, rely=0.4)
treshold_entry.insert(0, 15)

iou_label = ttk.Label(root, text="IOU (%):")
iou_label.place(relx=0.75, rely=0.4)

iou_entry = ttk.Entry(root, width=6)
iou_entry.place(relx=0.82, rely=0.4)
iou_entry.insert(0, 10)



#Run button
run_button = ttk.Button(root, text="Run detector",width=15, command=submit_form)
run_button.place(relx=0.06, rely=0.5)

#Clean button
clean_space_button = ttk.Button(root, text="Clean space",width=15, command=clean_space)
clean_space_button.place(relx=0.2, rely=0.5)

# Separator object
separator = ttk.Separator(root, orient='horizontal')
separator.place(relx=0, rely=0.6, relwidth=1, relheight=1)

# Load the image
image_path_gorilla = "images/Logo Dian fossey.png"  # Replace with your image file path
image1 = Image.open(image_path_gorilla)
image1 = image1.resize((100, 100))  # Resize the image if needed  cornell-bird-lab-1536x1152.jpg
photo1 = ImageTk.PhotoImage(image1)


image_path_cornell = "images/cornell-bird-lab-1536x1152.jpg"  # Replace with your image file path
image2 = Image.open(image_path_cornell)
image2 = image2.resize((80, 80))  # Resize the image if needed  cornell-bird-lab-1536x1152.jpg
photo2 = ImageTk.PhotoImage(image2)


# Add the image to a label
logo1_label = tk.Label(root, image=photo1)
logo1_label.place(relx=0.02, rely=0.7)  # Adjust padding as needed

# Add the image to a label
logo1_label = tk.Label(root, text='K. Lisa Yang Center for Conservation Bioacoustics')
logo1_label.place(relx=0.5, rely=0.8)  # Adjust padding as needed

# Add the image to a label
logo2_label = tk.Label(root, image=photo2)
logo2_label.place(relx=0.85, rely=0.7)  # Adjust padding as needed

# Add the image to a label
conceptor_label = tk.Label(root, text='Vainqueur BULAMBO')
conceptor_label.place(relx=0.02, rely=0.9)  # Adjust padding as needed


root.mainloop()





