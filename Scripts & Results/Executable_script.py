import tkinter as tk
from tkinter import messagebox
from BERT_and_Fuzzy_System_Implementation import predict_sentiment_fuzzy

# Function to analyze sentiment when the button is clicked
def analyze_sentiment():
    input_text = text_entry.get("1.0", "end-1c")
    if input_text:
        predicted_fuzzy_sentiment = predict_sentiment_fuzzy(input_text)
        result_label.config(text="Predicted Sentiment: " + predicted_fuzzy_sentiment)
    else:
        messagebox.showwarning("Input Error", "Please enter some text for analysis.")

# Create a tkinter window
window = tk.Tk()
window.title("Sentiment Analysis with Fuzzy Logic")

# Set the window size
window.geometry("800x600")  # Adjust the dimensions as needed

# Create a larger text entry field
text_entry = tk.Text(window, height=15, width=60, font=("Georgia", 16))  # Adjust font size and text box size
text_entry.pack()

# Create an analyze button
analyze_button = tk.Button(window, text="Analyze Sentiment", command=analyze_sentiment, font=("Arial", 14))
analyze_button.pack()

# Create a label to display the result
result_label = tk.Label(window, text="", font=("Arial", 16))
result_label.pack()

# Start the tkinter main loop
window.mainloop()
