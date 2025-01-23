import tkinter as tk
from tkinter import ttk, filedialog, messagebox

def load_file():
    """Load records from a text file and populate the table."""
    file_path = filedialog.askopenfilename(
        title="Select a Text File", filetypes=[("Text Files", "*.txt")]
    )
    if file_path:
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Clear existing table data
            for row in tree.get_children():
                tree.delete(row)

            # Populate table with file records
            for index, line in enumerate(lines):
                tree.insert("", "end", iid=index, values=(index, line.strip()))

            messagebox.showinfo("File Loaded", f"{len(lines)} records loaded into the table.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")

def on_record_select(event):
    """Display the selected record."""
    selected_item = tree.focus()  # Get the selected item (row ID)
    if selected_item:
        record = tree.item(selected_item)['values']  # Get the values of the selected row
        #record_label.config(text=f"Selected Record:\nIndex: {record[0]}\nContent: {record[1]}")
        print(f"Selected Item: {selected_item}")

# Create the Tkinter GUI
window = tk.Tk()
window.title("Populate Table from File")
window.geometry("600x400")

# Button to load the text file
load_button = tk.Button(window, text="Load Text File", command=load_file)
load_button.pack(pady=10)

# Treeview for displaying file records
columns = ("Index", "Content")
tree = ttk.Treeview(window, columns=columns, show="headings", height=15)
tree.heading("Index", text="Index")
tree.heading("Content", text="Content")
tree.column("Index", width=50, anchor="center")
tree.column("Content", width=500, anchor="w")
tree.pack(pady=10)

# Bind the treeview selection event
tree.bind("<<TreeviewSelect>>", on_record_select)

# Label to display the selected record
record_label = tk.Label(window, text="Select a record to view its details.", wraplength=500, justify="left")
record_label.pack(pady=10)

# Run the Tkinter event loop
window.mainloop()
