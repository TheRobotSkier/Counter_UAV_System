import librosa
import os

# Define the path to the audio file
audio_path = os.path.join('..', 'Recordings', 'Other_sounds', 'drilling and background sound.m4a')

# Load the audio file
y, sr = librosa.load(audio_path)

print(f"Sampling rate of the audio file: {sr} Hz")
print(f"Length of audio: {len(y) / sr:.2f} seconds")