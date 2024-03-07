import numpy as np
import keyboard 
from tensorflow.keras import models

from recording_helper import record_audio, terminate
from tensorflow_helper import preprocess_audiobuffer

# !! Modify this in the correct order
commands = ['up', 'left', 'stop', 'right', 'no', 'down', 'yes', 'go']

loaded_model = models.load_model("saved_cnn_model")

def predict_mic():
    audio = record_audio()
    spec = preprocess_audiobuffer(audio)
    prediction = loaded_model(spec)
    label_pred = np.argmax(prediction, axis=1)
    command = commands[label_pred[0]]
    print("Command:", command)
    return command

if __name__ == "__main__":
    print("\n===== Command Speech Recognition =====")
    print("Press 'Space' for speech recognition.")
    print("Press 'Esc' to terminate.")
    try:
        while True:
            if keyboard.is_pressed("space"):
                command = predict_mic()
            elif keyboard.is_pressed("esc"):
                print("\nThank you for using Command Speech Recognition!")
                terminate()
                break
    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
        terminate()