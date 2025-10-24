DRILLING_SOUND_1 = "Acoustics_system\Recordings\Other_sounds\drilling and background sound.wav"
import librosa

# Load the audio (librosa uses audioread under the hood)
y, sr = librosa.load(DRILLING_SOUND_1, sr=None)  # sr=None keeps original rate

print(f"Loaded audio with sampling rate: {sr} Hz")
print(f"Length of audio: {len(y) / sr:.2f} seconds")