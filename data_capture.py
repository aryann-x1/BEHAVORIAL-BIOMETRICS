from pynput import keyboard
import pandas as pd
import time

# Initialize an empty list to hold the data
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
    # Stop listener
    if key == keyboard.Key.esc:
        return False

# Collect events until released
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()

# Save captured data to CSV after listener stops
df = pd.DataFrame(data)
df.to_csv('captured_keystrokes.csv', index=False)
print("Data captured and saved to 'captured_keystrokes.csv'.")
