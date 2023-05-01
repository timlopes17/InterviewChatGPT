import openai
import json
import whisper
import pyaudio
import wave
import keyboard
from elevenlabslib import *

with open('config.json', 'r') as f:
    data = json.load(f)

GPTKey = data['GPTKey']
messages = [
    {"role": "system",
     "content": "I am conducting an interview and you are the guest. The interview is about large language machines like yourself."
                "Be polite and kind and answer the questions I ask the best way you can without being lengthy. Try to keep responses"
                "below 150 words"}
]

user = ElevenLabsUser(data['ElevenLabs'])
voice = user.get_voices_by_name("Bella")[0]

model = whisper.load_model("base")

# Set up the PyAudio stream
CHUNK_SIZE = 1024
SAMPLE_RATE = 44100
RECORD_SECONDS = 5

p = pyaudio.PyAudio()

openai.api_key = GPTKey

frames = []

# Define a callback function to handle audio recording
def callback(in_data, frame_count, time_info, status):
    frames.append(in_data)
    return in_data, pyaudio.paContinue


loops = 0

while True:
    # Wait for the "r" key to be pressed
    frames = []
    print("(Ready)")
    keyboard.wait('r')

    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE,
                    stream_callback=callback)

    stream.start_stream()
    print("recording...")

    keyboard.wait('r', lambda _: stream.stop_stream())

    stream.close()
    print("Stop recording")

    wave_file = wave.open('audio.wav', 'wb')
    wave_file.setnchannels(1)
    wave_file.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wave_file.setframerate(SAMPLE_RATE)
    wave_file.writeframes(b''.join(frames))
    wave_file.close()

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio('audio.wav')
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # decode the audio
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)

    # print the recognized text
    print(f"\nTim: {result.text}")

    messages.append({"role": "user", "content": result.text})

    chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    reply = chat.choices[0].message.content

    messages.append(
        {"role": 'assistant', 'content': reply}
    )

    print(f"\nChatGPT: {reply}")

    voice.generate_and_play_audio(reply, playInBackground=False)
