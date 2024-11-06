import pandas as pd
import numpy as np
import requests  # To send HTTP requests to Flask
from pynput import keyboard
import time

# Define the Flask server URL
FLASK_SERVER_URL = "http://127.0.0.1:5000/authenticate"  # Ensure this matches your Flask app's URL

# Function to capture real-time keystroke data
def capture_keystrokes():
    data = []

    def on_press(key):
        try:
            data.append({
                'key': key.char,
                'time': time.time()
            })
        except AttributeError:
            data.append({
                'key': str(key),
                'time': time.time()
            })

    def on_release(key):
        # Stop listener if Esc key is pressed
        if key == keyboard.Key.esc:
            return False

    # Start capturing keystrokes
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

    # Convert captured data to DataFrame
    df = pd.DataFrame(data)
    
    # Ensure enough keystrokes are captured for a sequence
    if len(df) < 10:  # Assuming you want a sequence length of 10
        print("Not enough keystrokes captured for authentication.")
        return None

    # Calculate time differences between consecutive keystrokes
    df['time_diff'] = df['time'].diff().fillna(0)  # Time difference between keystrokes

    # Select the most recent sequence of time differences for authentication
    recent_sequence = df['time_diff'].tail(10).values  # Get the last 10 time differences

    # Prepare the data for sending to the Flask backend
    return recent_sequence.tolist()  # Return as a list

# Main loop for real-time authentication
while True:
    print("Capturing keystrokes for authentication... (Press ESC after typing at least 10 keys)")

    # Capture keystrokes and extract features
    time_diffs = capture_keystrokes()

    # If no valid features were captured, skip this iteration
    if time_diffs is None:
        continue

    # Send the time differences to the Flask backend
    try:
        response = requests.post(FLASK_SERVER_URL, json={'timeDiffs': time_diffs})
        result = response.json()
        print("Authentication result:", result['status'])
    except Exception as e:
        print("Error sending data to Flask server:", e)
