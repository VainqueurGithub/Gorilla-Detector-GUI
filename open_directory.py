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