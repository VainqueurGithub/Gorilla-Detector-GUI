import tkinter as tk

def show_input():
    entered_text = user_input.get()  # Get the value of the Entry widget
    label.config(text=f"You entered: {entered_text}")

# Create the main application window
app = tk.Tk()
app.title("Default Value Example")
app.geometry("300x150")

# Create an Entry widget and insert a default value directly
user_input = tk.Entry(app, width=30)
user_input.insert(0, "Default Text")  # Insert default value at the beginning
user_input.pack(pady=10)

# Add a button to get the input value
submit_button = tk.Button(app, text="Submit", command=show_input)
submit_button.pack(pady=5)

# Add a label to display the entered text
label = tk.Label(app, text="")
label.pack()

# Run the application
app.mainloop()
