import enum
import subprocess
import statistics
import csv
from pathlib import Path
import importlib
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
from datetime import datetime
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
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
sns.set_theme(style="darkgrid")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#save
# settings
hop_length = 512 # number of samples per time-step in spectrogram
n_mels = 128 # number of bins in spectrogram. Height of image
time_steps = 384 # number of time-steps. Width of image
audio_duration = 6.0  # Total audio duration in seconds
max_freq = 8000.0  # Maximum frequency in Hz
image_width = 640  # Spectrogram width (time axis)
image_height = 320  # Spectrogram height (frequency axis)
min_freq =0
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

def predict_class(filename,conf, iou, show=False, imgsz=640,save=True,name='yolov8m'):
    
    model = YOLO('model/runs/detect/yolov8l_custom/weights/best.pt')
    
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

def spectrogram_image(y, sr, out, hop_length, n_mels, fmax=5000):
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
    decision = 'Underfined'
    i=0
    detector_table = pd.DataFrame(columns=['Selection', 'View', 'Channel', 'Begin Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)', 'label', 'conf', 'chrunk', 'decision'])
    #filename_chunk_spect
    time_elaspe = 0
    for j,c in enumerate(chunk_array):
        if c.endswith((".png", ".PNG")):
            result = predict_class(c, conf, iou)
            for r in result:
                for cl in r.boxes:
                    bbox = cl.xywhn
                    x_center, y_center, width, height  = bbox[0].tolist()  # Convert to individual scalar values
                    # Denormalize bounding box
                    x_min = (x_center - width / 2) * image_width
                    x_max = (x_center + width / 2) * image_width
                    y_min = (y_center - height / 2) * image_height
                    y_max = (y_center + height / 2) * image_height
                    if j==0:
                        time_start = (x_min / image_width) * audio_duration
                        time_end = (x_max / image_width) * audio_duration
                        freq_start = min_freq + (1 - y_max / image_height) * (max_freq - min_freq)
                        freq_end = min_freq + (1 - y_min / image_height) * (max_freq - min_freq)
                    else:
                        time_start = time_elaspe+((x_min / image_width) * audio_duration)
                        time_end = time_elaspe+((x_max / image_width) * audio_duration)
                        freq_start = min_freq + (1 - y_max / image_height) * (max_freq - min_freq)
                        freq_end = min_freq + (y_min / image_height) * (max_freq - min_freq)
                        
                    if freq_start < 0:
                           freq_start = freq_start * (-1)
                    detector_table.loc[i] = [Selection, View, Channel,time_start,time_end,freq_start,freq_end,label,cl.conf[0].item(),j+1,decision]
                    i+=1
                    Selection+=1
            time_elaspe +=6

    filename_chunk_spect = os.path.basename(c)
    filename_chunk_spect = filename_chunk_spect.split(".wav")
    output_file = output_path+'/'+filename_chunk_spect[0]+'_duration_'+str(duration_sec)+'.txt'
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

    cluster_detection_button = ttk.Button(margin_form_window, text="Clustering Analysis",command=clustering_form)
    cluster_detection_button.place(relx=0.67, rely=0.24, relheight=0.1, relwidth=0.3)

def metadata_read_csv(source_metadata, file):

    chemin_metadata = Path(source_metadata+'/'+file)

    if chemin_metadata.exists():
        # Open the file and read the first line to detect the delimiter
        with open(source_metadata+'/'+file, "r") as csvfile:
            # Read the file's content
            sample = csvfile.readline()
        
            # Use Sniffer to detect the delimiter
            dialect = csv.Sniffer().sniff(sample)
            delimiter = dialect.delimiter
    
        # Read the data to pandas dataframe by assigning the correct delimiter

        if delimiter==',':
            data = pd.read_csv(source_metadata+'/'+file, sep = ',', encoding = 'latin1')
        elif delimiter==';':
            data = pd.read_csv(source_metadata+'/'+file, sep = ';', encoding = 'latin1')
        return data
    else:
        messagebox.showerror("Error", "The meta data file is not  found in the output folder, or has been wrongly named. Make sure you have metadata_swifts.csv in output folder")

def map_detection_form():
    detection_frame = pd.DataFrame(columns=['Selection', 'View', 'Channel', 'Begin Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)', 'label', 'conf','hour_voc', 'date_voc','lat','long', 'swift', 'site', 'land_use', 'land_cover', 'vegetation'])
    gdf = pd.DataFrame(columns=['Selection', 'View', 'Channel', 'Begin Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)', 'label', 'conf','hour_voc', 'date_voc','lat','long', 'swift', 'site', 'land_use', 'land_cover', 'vegetation'])
    folder_path = select_output_folder()

    metadata = metadata_read_csv(folder_path, 'metadata_swifts.csv')

    try:
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".txt"):
                data_frame = pd.read_csv(folder_path+'/'+file_name, sep='\t')

                file_metadata = file_name.split('_duration_')
                file_metadata = file_metadata[0].split('_')

                swift = metadata.loc[metadata['swift']== file_metadata[0]]
                data_frame['lat'] = swift.loc[swift.index[0],'lat']
                data_frame['long'] = swift.loc[swift.index[0],'long']
                data_frame['swift'] = swift.loc[swift.index[0],'swift']
                data_frame['site'] = swift.loc[swift.index[0],'site']
                data_frame['land_use'] = swift.loc[swift.index[0],'land_use']
                data_frame['land_cover'] = swift.loc[swift.index[0],'land_cover']
                data_frame['vegetation'] = swift.loc[swift.index[0],'vegetation']

                detection_frame = pd.concat([detection_frame, data_frame], axis=0, ignore_index=True) # concatenating along rows
                geometry = [Point(xy) for xy in zip(round(metadata['lat']),round(metadata['long']))]
                gdf = gpd.GeoDataFrame(metadata, geometry=geometry) 
    except Exception as e:
        print(e) 
    
    current_directory = os.getcwd()
    shape_directory = current_directory+'/plots/map'
    counts_df = detection_frame['swift'].value_counts().reset_index()
    counts_df.columns = ['swift', 'counts']
    counts_df
    for file in os.listdir(shape_directory):
        # check the files which are end with specific extension
	    if file.endswith(".shp"):
              data = gpd.read_file(os.path.join(shape_directory, file))
              if gdf.empty:
                  data.plot(color="lightgrey", edgecolor="black", alpha=0.7)
                  plt.title("Detection distribution Map.")
                  plt.show()

              gdf['events'] = metadata['swift'].map(counts_df.set_index('swift')['counts'])
              gdf.plot(ax = data.plot(color="lightgrey", edgecolor="black", alpha=0.7), markersize=gdf['events'] * 5, alpha=0.6, color='red', edgecolor='black')

              for x,y, label in zip(gdf.geometry.x, gdf.geometry.y, gdf['swift']):
                plt.text(x,y, label, fontsize=12, ha='right', va='bottom', color='blue')
              plt.title("Detection distribution Map.")
              plt.show()
              

def visualization_form():
    detection_frame = pd.DataFrame(columns=['Selection', 'View', 'Channel', 'Begin Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)', 'label', 'conf','hour_voc', 'date_voc','lat','long', 'swift', 'site', 'land_use', 'land_cover', 'vegetation'])
    folder_path = select_output_folder()

    try:
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".txt"):

                data_frame = pd.read_csv(folder_path+'/'+file_name, sep='\t')
                file_name = file_name.split('_duration_')
                file_name = file_name[0].split('_')

                parsed_date = datetime.strptime(file_name[1], "%Y%m%d").date()
                parsed_time = datetime.strptime(file_name[2], "%H%M%S").time()

                data_frame['hour_voc'] = parsed_time
                data_frame['date_voc'] = parsed_date

                metadata = metadata_read_csv(folder_path, 'metadata_swifts.csv')
                swift = metadata.loc[metadata['swift']== file_name[0]]
                data_frame['lat'] = swift.loc[swift.index[0],'lat']
                data_frame['long'] = swift.loc[swift.index[0],'long']
                data_frame['swift'] = swift.loc[swift.index[0],'swift']
                data_frame['site'] = swift.loc[swift.index[0],'site']
                data_frame['land_use'] = swift.loc[swift.index[0],'land_use']
                data_frame['land_cover'] = swift.loc[swift.index[0],'land_cover']
                data_frame['vegetation'] = swift.loc[swift.index[0],'vegetation']
                detection_frame = pd.concat([detection_frame, data_frame], axis=0, ignore_index=True) # concatenating along rows
    except Exception as e:
        print(e) 

    detection_frame['date_voc'] = pd.to_datetime(detection_frame.date_voc)
    detection_frame['year'] = detection_frame['date_voc'].dt.year
    detection_frame['month'] = detection_frame['date_voc'].dt.month
    detection_frame['day'] = detection_frame['date_voc'].dt.day
    detection_frame['day_name'] = detection_frame['date_voc'].dt.day_name()
    
    # Extract month name using apply
    detection_frame['month_name'] = detection_frame['date_voc'].apply(lambda x: x.strftime('%B'))

    #detection_frame['hour_voc'] = pd.to_datetime(detection_frame.hour_voc)
    detection_frame['hour'] = detection_frame['hour_voc'].apply(lambda x: x.hour)
    
    #detection_frame.to_csv('output1.csv', index=False)
    
    data_plot = detection_frame.value_counts(['site', 'month_name', 'hour']).reset_index().rename(columns={0:'count'})
    #detection_frame.set_index(pd.DatetimeIndex(detection_frame['date_voc']), inplace=True)

    df_pivot = data_plot.pivot_table(index=['month_name', 'site'], columns='hour', values='count', aggfunc='sum').fillna(0).reset_index()

    # Melt for Seaborn compatibility
    df_melted = df_pivot.melt(id_vars=['month_name', 'site'], var_name='hour', value_name='count')

    try:
        # Create FacetGrid for stacking within each 'Site' facet
        g = sns.FacetGrid(df_melted, col="site", height=5, aspect=1, col_wrap=4)
        g.map_dataframe(sns.barplot, x="month_name", y="count", hue="hour", dodge=False, palette='dark:#4c72b0')
        g.map_dataframe(sns.barplot, x="month_name", y="count", hue="hour", dodge=False, palette='light:#4c72b0')

        # Adjust labels and title
        g.set_axis_labels("month_name", "count")
        g.set_titles("{col_name}")
        g.add_legend(title="hour")

        sns.catplot(data=detection_frame, x="label", y="hour", hue="month_name", kind="swarm", col="site", aspect=.7,)
        g = sns.catplot(data=detection_frame,x="label", y="month_name", row="site", kind="box", orient="h", sharex=False, margin_titles=True, height=1.5, aspect=4,)
    
        g.set(xlabel="Detections", ylabel="")
        g.set_titles(row_template="{row_name}")
        for ax in g.axes.flat:
            ax.xaxis.set_major_formatter('')

        fig, axes = plt.subplots(2,2, figsize=(17,5))
        lineplot = sns.lineplot(x="hour", y="conf",hue="month_name",data=detection_frame, ax=axes[0,0])
        lineplot.axes.set_title('Detection confidence over time')
        lineplot = sns.lineplot(x="hour", y="High Freq (Hz)",hue="month_name",data=detection_frame, ax=axes[0,1])
        lineplot.axes.set_title('Frequency Distribution over time')


        lineplot = sns.histplot(data=detection_frame, x="hour", ax=axes[1,0])
        lineplot.axes.set_title('Hourly detection')
        lineplot = sns.histplot(data=detection_frame, x="month_name", ax=axes[1,1])
        lineplot.axes.set_title('Monthy Detection')
        plt.show()

    except Exception as e:
        print(e)

def clustering_form():
    folder_path = select_output_folder()
    detection_frame = pd.DataFrame(columns=['Begin Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)', 'label', 'conf','hour_voc', 'date_voc','land_use', 'land_cover', 'vegetation'])

    try:
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".txt"):

                data_frame = pd.read_csv(folder_path+'/'+file_name, sep='\t')
                file_name = file_name.split('_duration_')
                file_name = file_name[0].split('_')

                parsed_date = datetime.strptime(file_name[1], "%Y%m%d").date()
                parsed_time = datetime.strptime(file_name[2], "%H%M%S").time()

                data_frame['hour_voc'] = parsed_time
                data_frame['date_voc'] = parsed_date

                metadata = metadata_read_csv(folder_path, 'metadata_swifts.csv')
                swift = metadata.loc[metadata['swift']== file_name[0]]
                data_frame['land_use'] = swift.loc[swift.index[0],'land_use']
                data_frame['land_cover'] = swift.loc[swift.index[0],'land_cover']
                data_frame['vegetation'] = swift.loc[swift.index[0],'vegetation']
                detection_frame = pd.concat([detection_frame, data_frame], axis=0, ignore_index=True) # concatenating along rows
    except Exception as e:
        print(e)

    detection_frame['date_voc'] = pd.to_datetime(detection_frame.date_voc)
    detection_frame['month'] = detection_frame['date_voc'].dt.month
    detection_frame['day'] = detection_frame['date_voc'].dt.day
    detection_frame['day_name'] = detection_frame['date_voc'].dt.day_name()
    
    # Extract month name using apply
    detection_frame['month_name'] = detection_frame['date_voc'].apply(lambda x: x.strftime('%B'))

    #detection_frame['hour_voc'] = pd.to_datetime(detection_frame.hour_voc)
    detection_frame['hour'] = detection_frame['hour_voc'].apply(lambda x: x.hour)

    # Label Encode 'environment'
    label_encoder = LabelEncoder()
    #detection_frame["label_encoded"] = label_encoder.fit_transform(detection_frame["label"])
    detection_frame["land_use_encoded"] = label_encoder.fit_transform(detection_frame["land_use"])
    detection_frame["land_cover_encoded"] = label_encoder.fit_transform(detection_frame["land_cover"])
    detection_frame["vegetation_encoded"] = label_encoder.fit_transform(detection_frame["vegetation"])
    # Combine numerical and encoded categorical data
   # data_final = pd.concat([detection_frame.drop(["vocalization_type", "environment"], axis=1), encoded_df], axis=1)

    # Convert to NumPy matrix
    data_final = detection_frame[['Begin Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)', 'conf','land_use_encoded', 'land_cover_encoded', 'vegetation_encoded', 'month','hour','day']]

    try:
        # Apply K-Means Clustering
        kmeans = KMeans(n_clusters=2, random_state=42)
        data_final['cluster'] = kmeans.fit_predict(data_final)

        # Scatter Plot with Cluster Labels
        plt.figure(figsize=(8, 5))

        # Plot each data point
        for i in range(len(data_final)):
            plt.scatter(data_final["Begin Time (s)"][i], data_final["High Freq (Hz)"][i], c=f"C{data_final['cluster'][i]}", label=f"Cluster {data_final['cluster'][i]}" if i == 0 else "", edgecolors="k")
            plt.text(data_final["End Time (s)"][i], data_final["High Freq (Hz)"][i], detection_frame["vegetation"][i], fontsize=10, ha='right', color="black")

        # Label Axes
        plt.xlabel("End Time (s)")
        plt.ylabel("High Frequency (Hz)")
        plt.title("Vocalization Clusters with Labels")
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(e)

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
    try:
       # Import the script where the function is defined
        script_name = "detection"  # Name of the Python file without the .py extension
        module = importlib.import_module(script_name)
        
        # Call the function by its name
        function_name = "open_table_window"  # Name of the function to call
        func = getattr(module, function_name)  # Get the function reference
        func()  # Call the function
    except AttributeError:
        print(f"Error: Function '{function_name}' not found in '{script_name}.py'.")
    except ModuleNotFoundError:
        print(f"Error: Script '{script_name}.py' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
   
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
#root.resizable(0, 0)

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





