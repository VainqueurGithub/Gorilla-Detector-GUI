import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from tkinter import ttk

def submit_form():
    # Retrieve data from input fields
    name = name_entry.get()
    email = email_entry.get()
    age = age_entry.get()
    
    # Basic validation
    if not name or not email or not age:
        messagebox.showerror("Error", "All fields are required!")
        return
    
    if not age.isdigit():
        messagebox.showerror("Error", "Age must be a number!")
        return
    
    # Display user data (can be saved to a file or database)
    messagebox.showinfo("Form Submitted", f"Name: {name}\nEmail: {email}\nAge: {age}")
    # Clear the form
    name_entry.delete(0, tk.END)
    email_entry.delete(0, tk.END)
    age_entry.delete(0, tk.END)

# Adjust the alignment of the grid
app.columnconfigure(0, weight=1)  # Make the first column adjustable
app.columnconfigure(1, weight=3)  # Make the second column adjustable

# Create the main application window
app = tk.Tk()
label = tk.Label(app, text="Welcome to Gorilla detector", font=("Arial", 16), fg="blue")
#label.grid(column=0, row=1, sticky=tk.W, padx=5, pady=5)
app.title("Gorilla Detector")
app.geometry("800x500")
app.resizable(0, 0)


# Load the image
image_path = "Logo Dian fossey.png"  # Replace with your image file path
image = Image.open(image_path)
image = image.resize((150, 150))  # Resize the image if needed
photo = ImageTk.PhotoImage(image)

# Add the image to a label
logo_label = ttk.Label(app, image=photo)
logo_label.grid(column=1, row=1, sticky=tk.E, padx=5, pady=5)  # Adjust padding as needed


# Add form labels and input fields
#tk.Label(app, text="Name:").pack(pady=5)
#name_entry = tk.Entry(app, width=30)
#name_entry.pack()

#tk.Label(app, text="Email:").pack(pady=5)
#email_entry = tk.Entry(app, width=30)
#email_entry.pack()

#tk.Label(app, text="Age:").pack(pady=5)
#age_entry = tk.Entry(app, width=30)
#age_entry.pack()

# Add a submit button
#submit_button = tk.Button(app, text="Submit", command=submit_form)
#submit_button.pack(pady=20)


# Start the application
app.mainloop()
