import tkinter as tk
from tkinter import ttk
import geopandas as gpd
import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg") # for backend
from shapely.geometry import Point
import seaborn as sns
import importlib
from tkinter import messagebox, filedialog

open_directory = "open_directory"
open_directory = importlib.import_module(open_directory)



def map_detection_form():
    detection_frame = pd.DataFrame(columns=['Selection', 'View', 'Channel', 'Begin Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)', 'label', 'conf','hour_voc', 'date_voc','lat','long'])
    gdf = pd.DataFrame(columns=['Selection', 'View', 'Channel', 'Begin Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)', 'label', 'conf','hour_voc', 'date_voc','lat','long'])
    folder_path = open_directory.select_output_folder()

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
    folder_path = open_directory.select_output_folder()

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
    folder_path = open_directory.select_output_folder()

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










# Create a new window (child window)
margin_form_window = tk.Tk()
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

margin_form_window.mainloop()