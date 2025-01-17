import tkinter as tk
from tkinter import ttk
from unittest import TestResult
from PIL import Image, ImageTk
from tkinter import messagebox, filedialog
from tkinter import PhotoImage
import time
import threading

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

detector_table = pd.DataFrame(columns=['Selection', 'View', 'Channel', 'Begin Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)', 'label', 'conf'])

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

    # Resize to YOLO's expected size (e.g., 640x640)
    #image_resized = cv2.resize(img, (640, 640))
    #tensor_input = torch.from_numpy(image_resized).unsqueeze(0).unsqueeze(0).float() / 255.0
    # Expand grayscale to RGB format
    #tensor_input_rgb = tensor_input.expand(-1, 3, -1, -1)  # Shape: (1, 3, 640, 640)
    # Save as PNG
    skimage.io.imsave(out, img)
    return img


def main_function(hop_length, n_mels, time_steps, input_path, path_audio_chunks):
    directory_chunk = os.fsencode(path_audio_chunks)
    directory_input = os.fsencode(input_path)
    max_value = count_wav_files(input_path)
    j=0
    for input_file in os.listdir(directory_input):
        filename = os.fsdecode(input_file)
        max_value = count_wav_files(path_audio_chunks)
        start_task(max_value,j)
        if filename.endswith((".wav", ".WAV")):
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
                    # convert to PNG
                    img = spectrogram_image(window, sr=sr, out=out, hop_length=hop_length, n_mels=n_mels)
                    j+=1
                    continue
                else:
                    continue



















def detection_out_put(input_path,path_audio_chunks,conf, iou, output_path,path_audio_chunk_spect,detector_table,audio_duration,max_frequency,View,Channel,label):
    directory_chunk_spect = os.fsencode(path_audio_chunk_spect)
    Selection=1
    View = 'Spectrogram 1'
    Channel = 1
    label='Gorilla'
    i=0
    j=0
    main_function(hop_length, n_mels, time_steps, input_path, path_audio_chunks)
    max_value = count_spect_files(path_audio_chunk_spect)
    
    for file in os.listdir(directory_chunk_spect):
        filename_chunk_spect = os.fsdecode(file)
        start_task(max_value,j)
        if filename_chunk_spect.endswith((".png", ".PNG")):
            result = predict_class(path_audio_chunk_spect+'/'+filename_chunk_spect, conf, iou)
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
                    j+=1
                output_file = output_path+'/'+filename_chunk_spect+'.txt'
                detector_table.to_csv(output_file, sep="\t", index=False)


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
                    detection_out_put(input_path,path_audio_chunks,conf, iou, output_path,path_audio_chunk_spect,detector_table,audio_duration,max_frequency,View,Channel,label)
                else:
                    messagebox.showerror("Error", "IOU must be between 0 and 100!")
            else:
                messagebox.showerror("Error", "IOU must be a number!")
        else:
            messagebox.showerror("Error", "Treshold must be between 0 and 100!")
    else:     
        messagebox.showerror("Error", "Treshold must be a number!")
     
        
    

def start_task(max_value, i):
    # Disable the Start button
    run_button.config(state="disabled")
    # Open the popup window with the progress bar
    popup_progress()

    # Run the task in a separate thread to avoid blocking the main GUI
    threading.Thread(target=simulate_task(max_value, i)).start()

def popup_progress():
    # Create a new popup window
    global progress_popup, progress_bar, progress_label
    progress_popup = tk.Toplevel(root)
    progress_popup.title("Progress")
    progress_popup.geometry("300x100")
    progress_popup.resizable(False, False)

    # Add a label
    progress_label = tk.Label(progress_popup, text="Processing...")
    progress_label.grid(column=1, row=10, sticky=tk.E, padx=5, pady=5)

    # Add the progress bar
    progress_bar = ttk.Progressbar(progress_popup, orient="horizontal", length=250, mode="determinate")
    progress_bar.grid(column=1, row=10, sticky=tk.E, padx=5, pady=5)

def simulate_task(max_value, i):
    # Simulate a long-running task
    #max_value = 100
    progress_bar["value"] = 0
    progress_bar["maximum"] = max_value

    while i <= max_value + 1:
        i += 1
    #for i in range(max_value + 1):
        time.sleep(0.05)  # Simulate work by sleeping
        progress_bar["value"] = i
        progress_label.config(text=f"Progress: {i}%")
        progress_popup.update_idletasks()  # Refresh the popup

    # Close the popup when the task is done
    progress_label.config(text="Task Complete!")
    time.sleep(1)
    progress_popup.destroy()
    run_button.config(state="normal")  # Re-enable the Start button


# CREATE MENU BUTTONS FUNCTIONS

# OUTPUT RESULT TEXT
def open_summary_result_window():
    # Create a new window (child window)
    new_window = tk.Toplevel(root)
    new_window.title("New Window")
    new_window.geometry("520x300")
    
    # Add content to the new window
    label = tk.Label(new_window, text="This is a new window!", font=("Arial", 12))
    label.pack(pady=20)
    
    # Add a button to close the new window
    close_button = tk.Button(new_window, text="Close", command=new_window.destroy)
    close_button.pack(pady=10)









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

open_button = tk.Button(root, text="Detections", command=open_summary_result_window)
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

show_detection_label = tk.Label(root, text="Save model output")
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





