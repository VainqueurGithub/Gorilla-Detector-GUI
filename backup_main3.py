import tkinter as tk
from tkinter import ttk
from unittest import TestResult
from PIL import Image, ImageTk
from tkinter import messagebox, filedialog
from tkinter import PhotoImage
import time
import threading
import random

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
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#save
# settings
hop_length = 512 # number of samples per time-step in spectrogram
n_mels = 128 # number of bins in spectrogram. Height of image
time_steps = 384 # number of time-steps. Width of image
spectrogram_width = 230  # Width of the spectrogram image
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


def predict_class(filename,conf, iou, show=False, imgsz=640,save=False,name='yolov8m'):
    model = YOLO('model/runs/detect/yolov8m_custom3/weights/best.pt')

    #Predicting
    result=model.predict(
        source=filename,show=show,imgsz=imgsz,save=save,name=name,conf=conf,iou=iou)
    return result

# Load the audio file

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

    # Export each chunk as a separate audio file
    for i, chunk in enumerate(chunks):
        chunk_name = f"{output_dir}/{filename}_chunk_{i+1}.wav"  # Change extension if needed
        chunk.export(chunk_name, format="wav")  # Export as WAV format

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
        chunk_spectrogram(filename,input_path,path_audio_chunks,directory_chunk,conf,iou,output_path)
        time.sleep(0.01 + (0.01 * (i / max_value)))
        progress["value"] = i
        label.config(text=f"Processing {i}/{len(files)}")
        root.update_idletasks()  # Refresh the GUI

    # Close the popup when the task is done
    label.config(text="Detection completed!")
    popup.after(1000, popup.destroy)
    run_button.config(state="normal")


def detection_out_put(filename_chunk_spect,conf, iou, output_path):
    Selection=1
    View = 'Spectrogram 1'
    Channel = 1
    label='Gorilla'
    i=0
    if filename_chunk_spect.endswith((".png", ".PNG")):
        result = predict_class(filename_chunk_spect, conf, iou)
        detector_table = pd.DataFrame(columns=['Selection', 'View', 'Channel', 'Begin Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)', 'label', 'conf'])
        for r in result:
            for cl in r.boxes:
                bbox = cl.xywhn
                x, y, width, height = bbox[0].tolist()  # Convert to individual scalar values
                # Denormalize bounding box
                x_min = (x - width / 2) * 640
                x_max = (x + width / 2) * 640
                y_min = (y - height / 2) * 384
                y_max = (y + height / 2) * 384
        
                # Convert to time and frequency ranges
                time_start = (x_min / 640) * audio_duration
                time_end = (x_max / 640) * audio_duration
                freq_start = (y_min / 384) * max_frequency
                freq_end = (y_max / 384) * max_frequency
                detector_table.loc[i] = [Selection, View, Channel,time_start,time_end,freq_start,freq_end,label,cl.conf[0].item()]
                i+=1
                Selection+=1
            filename_chunk_spect = os.path.basename(filename_chunk_spect)
            output_file = output_path+'/'+filename_chunk_spect+'.txt'
            detector_table.to_csv(output_file, sep="\t", index=False)


def chunk_spectrogram(filename,input_path,path_audio_chunks,directory_chunk,conf,iou,output_path,hop_length=hop_length, n_mels=n_mels, time_steps=time_steps):
    if filename.endswith((".wav", ".WAV")):
        time.sleep(0.1)  # Simulate processing time
        #split de entire into 6 secs clips
        audio_chunks(path_audio_chunks,input_path,filename)
        for chunk_file in os.listdir(directory_chunk):
            chunk_name = os.fsdecode(chunk_file)
            if chunk_name.endswith((".wav", ".WAV")):
                y, sr = librosa.load(path_audio_chunks+'/'+chunk_name, duration=10.0, sr=22050)
                out =path_audio_chunks+'/chunk_spect/'+chunk_name+'.png'
                # extract a fixed length window
                start_sample = 0 # starting at beginning
                length_samples = time_steps*hop_length
                window = y[start_sample:start_sample+length_samples]
                # Generate image and convert to PNG
                img = spectrogram_image(window, sr=sr, out=out, hop_length=hop_length, n_mels=n_mels)
                #Use the image (spectogram) for detection
                detection_out_put(out,conf, iou, output_path)
                continue
            else:
                continue


# CREATE MENU BUTTONS FUNCTIONS

# OUTPUT RESULT TEXT
def open_summary_result_window():
    # Create a new window (child window)
    summary_result_window = tk.Toplevel(root)
    summary_result_window.title("New Window")
    summary_result_window.geometry("700x350")
    
    # Treeview table to display files
    columns = ("Chunk","Detections","Size (bytes)", "Path")
    file_table = ttk.Treeview(summary_result_window, columns=columns, show="headings", height=15)
    file_table.heading("Chunk", text="Chunk")
    file_table.heading("Detections", text="Detections")
    file_table.heading("Size (bytes)", text="Size (bytes)")
    file_table.heading("Path", text="Path")
    file_table.column("Chunk", anchor="e", width=400)
    file_table.column("Detections", anchor="e", width=100)
    file_table.column("Size (bytes)", anchor="e", width=100)
    file_table.column("Path", anchor="e", width=100)
    file_table.pack(pady=10, fill="both", expand=True)

    # Scrollbar for the table
    scrollbar = ttk.Scrollbar(summary_result_window, orient="vertical", command=file_table.yview)
    file_table.configure(yscroll=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    file_table.bind("<Double-1>", on_item_click)
    return file_table

def select_output_folder():
    """Open a dialog to select a folder and list its files."""
    folder_path = filedialog.askdirectory(title="Select Folder")
    if folder_path:
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
            file_detection = lines-1
            file_size = os.path.getsize(file_path)
            file_Path = folder_path
            file_table.insert("", "end", values=(file_name, file_detection, file_size, file_Path))
    except Exception as e:
        print(e)

def open_file(file_path):
    """Open the selected file using the default system application."""
    try:
        if os.name == 'nt':  # For Windows
            os.startfile(file_path)
        else:  # For Unix-based systems (Linux, macOS)
            webbrowser.open(file_path)
    except Exception as e:
        print(e)

def on_item_click(event):
    """Handle click event on a Treeview item."""
    file_table = open_summary_result_window()
    selected_item = file_table.selection()
    if selected_item:
        file_path = file_table.item(selected_item[0])['values'][3]
        if file_path and os.path.isfile(file_path):
            open_file(file_path)


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
logo1_label.grid(column=0, row=12, sticky=tk.SW, padx=5, pady=5)  # Adjust padding as needed

open_button = tk.Button(root, text="Detections", command=select_output_folder)
open_button.grid(column=1, row=1, sticky=tk.N, padx=5, pady=5)

label_label3 = ttk.Label(root, text="", font=("Arial", 16))
label_label3.grid(column=0, row=2, sticky=tk.N, padx=5, pady=5)
label_label4 = ttk.Label(root, text="", font=("Arial", 16))
label_label4.grid(column=0, row=3, sticky=tk.N, padx=5, pady=5)



label_label1 = ttk.Label(root, text="", font=("Arial", 16))
label_label1.grid(column=0, row=11, sticky=tk.N, padx=5, pady=5)
label_label2 = ttk.Label(root, text="", font=("Arial", 16))
label_label2.grid(column=0, row=12, sticky=tk.N, padx=5, pady=5)
#logo2_label = tk.Label(root, image=photo2)
#logo2_label.grid(column=0, row=12, sticky=tk.S, padx=5, pady=5)  # Adjust padding as needed

# Form

# Add a button to trigger the folder selection dialog
select_input_folder_button = tk.Button(root, text="Select Input Folder", command=select_folder_input, width=20)
select_input_folder_button.grid(column=0, row=4, sticky=tk.W, padx=5, pady=5)

# Add a label to display the selected folder path
folder_input_path_label = tk.Label(root, text="No folder selected", wraplength=400)
folder_input_path_label.grid(column=1, row=4, sticky=tk.E, padx=5, pady=5)

# Add a button to trigger the folder selection dialog
select_output_folder_button = tk.Button(root, text="Select Output Folder", command=select_folder_output, width=20)
select_output_folder_button.grid(column=0, row=5, sticky=tk.W, padx=5, pady=5)

# Add a label to display the selected folder path
folder_output_path_label = tk.Label(root, text="No folder selected", wraplength=400)
folder_output_path_label.grid(column=1, row=5, sticky=tk.E, padx=5, pady=5)

# Add a checkbox
save_model_output_label = tk.Label(root, text="Save model output")
save_model_output_label.grid(column=0, row=6, sticky=tk.W, padx=5, pady=5)
checkbox1 = tk.Checkbutton(root, variable=save_model_output_var)
checkbox1.grid(column=0, row=6, sticky=tk.E, padx=5, pady=5)

show_detection_label = tk.Label(root, text="Show detection")
show_detection_label.grid(column=0, row=7, sticky=tk.W, padx=5, pady=5)
checkbox2 = tk.Checkbutton(root, variable=show_detection_var)
checkbox2.grid(column=0, row=7, sticky=tk.E, padx=5, pady=5)


# Label to display the status

treshold_label = ttk.Label(root, text="Treshold (%):")
treshold_label.grid(column=0, row=8, sticky=tk.W, padx=5, pady=5)

treshold_entry = ttk.Entry(root)
treshold_entry.grid(column=0, row=8, sticky=tk.E, padx=5, pady=5)
treshold_entry.insert(0, 20)

iou_label = ttk.Label(root, text="IOU (%):")
iou_label.grid(column=0, row=9, sticky=tk.W, padx=5, pady=5)

iou_entry = ttk.Entry(root)
iou_entry.grid(column=0, row=9, sticky=tk.E, padx=5, pady=5)
iou_entry.insert(0, 10)



#login button
run_button = ttk.Button(root, text="Run detector",width=15, command=submit_form)
run_button.grid(column=0, row=10, sticky=tk.E, padx=5, pady=5)

root.mainloop()





