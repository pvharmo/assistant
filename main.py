# Imports
import pyaudio
import numpy as np
from openwakeword.model import Model
import speech_recognition as sr
import httpx
import io
import json
import wave
import gtts
from pydub import AudioSegment

listening_audio = 'listening_indicator_2.wav'
sending_audio = 'listening_indicator.wav'

# Get microphone stream
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280
audio = pyaudio.PyAudio()
mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

owwModel = Model(enable_speex_noise_suppression=True, vad_threshold=0.6)

n_models = len(owwModel.models.keys())

s = sr.Microphone()
r = sr.Recognizer()
with s as source:
    r.adjust_for_ambient_noise(source)

def play_audio(file):
    with wave.open(file, 'rb') as wf:
        p = pyaudio.PyAudio()

        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                            channels=wf.getnchannels(),
                            rate=wf.getframerate(),
                            output=True)

        while len(data := wf.readframes(CHUNK)):
            stream.write(data)

        stream.close()

        p.terminate()

def listen_for_wake_word():
    wait_frames = 0
    while True:
        # Get audio
        audio = np.frombuffer(mic_stream.read(CHUNK), dtype=np.int16)

        # Feed to openWakeWord model
        prediction = owwModel.predict(audio)

        if wait_frames < 10:
            wait_frames += 1
            continue

        res = owwModel.prediction_buffer["hey_jarvis"][-1]

        if res > 0.6:
            return

# Run capture loop continuosly, checking for wakewords
if __name__ == "__main__":

    while True:
        listen_for_wake_word()

        print("Received wake word! Waiting for user to stop speaking")

        play_audio(listening_audio)

        with s as source:
            audio = r.listen(source)

        audio_file = io.BytesIO()
        audio_file.write(audio.get_wav_data())

        play_audio(sending_audio)

        res = httpx.post(
            "https://api.deepinfra.com/v1/inference/openai/whisper-large",
            headers={"Authorization": "bearer " + env.openai_key},
            files={"audio": audio_file}
        )

        text = res.json()["text"]
        print("User: ", text)

        res = httpx.post(
            "https://api.deepinfra.com/v1/openai/chat/completions",
            headers={"Authorization": "bearer " + env.openai_key, "Content-Type": "application/json"},
            json={
                "model": "Qwen/Qwen2-72B-Instruct",
                "messages": [
                    {"role": "system", "content": "You are an assistant to help with tasks. Kepp your answers short and to the point."},
                    {"role": "user", "content": text}
                ]
            }
        )

        message = res.json()["choices"][0]["message"]["content"]

        print(message)

        tts = gtts.gTTS(message, tld="ca", slow=False, lang="en")
        audio = tts.stream()

        mp3_fp = io.BytesIO()

        tts.write_to_fp(mp3_fp)
        # mp3_fp = open("response.mp3", "rb")
        mp3_fp.seek(0)
        audio = AudioSegment.from_mp3(mp3_fp)

        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)

        with wave.open(wav_io, 'rb') as wav_file:
            sample_width = int(wav_file.getsampwidth() * 1.2)
            channels = wav_file.getnchannels()
            framerate = int(wav_file.getframerate() * 1.2)
            frames = wav_file.readframes(wav_file.getnframes())


        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(sample_width),
                        channels=channels,
                        rate=framerate,
                        output=True)

        stream.write(frames)

        stream.stop_stream()
        stream.close()

        p.terminate()
