import librosa
import numpy as np

# Load audio
y, sr = librosa.load("Acoustics_system\Dataset_DronePoint\DS1\ExperimentallyCollected\Spark_Test_File1.wav", sr=16000)

# Preprocess
y = librosa.util.normalize(y)

# Feature extraction
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
zcr = librosa.feature.zero_crossing_rate(y)
rms = librosa.feature.rms(y=y)

# Aggregate (mean + std)
features = np.hstack([
    np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
    np.mean(spec_centroid), np.std(spec_centroid),
    np.mean(spec_bandwidth), np.std(spec_bandwidth),
    np.mean(zcr), np.std(zcr),
    np.mean(rms), np.std(rms)
])
print("Extracted features shape:", features.shape)
print("Feature vector:", features)