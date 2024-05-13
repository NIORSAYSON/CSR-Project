import streamlit as st
import numpy as np
import keyboard 
import pyaudio
from tensorflow.keras import models

from tensorflow_helper import preprocess_audiobuffer

# !! Modify this in the correct order
commands = ['up', 'left', 'stop', 'right', 'no', 'down', 'yes', 'go']

loaded_model = models.load_model("saved_cnn_model")

st.sidebar.header('Audio Parameters')

FRAMES_PER_BUFFER = int(st.sidebar.text_input('Frames per buffer', 3200))
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = int(st.sidebar.text_input('Rate', 16000))
p = pyaudio.PyAudio()
st.markdown(
    """
    <div style='background-color: blue; padding: 10px; border-radius: 5px;'>
        <h2 style='color: white; text-align: center;'>Command Speech Recognition 🎤</h2>
    </div>
    """,
    unsafe_allow_html=True
)
with st.expander('About this App'):
    st.markdown(
    """
    <div style='background-color: orange; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
        <p style='color: black; text-align: center; font-weight: bold;'>Press the 'Start' button to begin speech recognition.</p>
        <p style='color: black; text-align: center; font-weight: bold;'>Press the 'Stop' button to terminate the application.</p>
    </div>
    """,
    unsafe_allow_html=True
)

def predict_mic():
    audio = record_audio()
    spec = preprocess_audiobuffer(audio) 
    prediction = loaded_model(spec)
    label_pred = np.argmax(prediction, axis=1)
    command = commands[label_pred[0]]
    st.info("Command: {}".format(command))
    return command

def record_audio():
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER
    )

    st.write("\nStart Recording...")

    frames = []
    seconds = 2
    for i in range(0, int(RATE / FRAMES_PER_BUFFER * seconds)):
        data = stream.read(FRAMES_PER_BUFFER)
        frames.append(data)

    stream.stop_stream()
    stream.close()

    return np.frombuffer(b''.join(frames), dtype=np.int16)

def start_listening():
	st.session_state['run'] = True

def stop_listening():
	st.session_state['run'] = False
     
def terminate():
    st.success("Thank you for using Command Speech Recognition!")
    p.terminate()

if __name__ == "__main__":
    col1, col2 = st.columns(2)

    if col1.button('Start'):
         predict_mic()
    if col2.button('Stop', type="primary"):
         terminate()