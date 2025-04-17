import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

def load_audio(file_path, sr=16000):
    y, _ = librosa.load(file_path, sr=sr)
    if len(y) == 0:
        raise ValueError("Empty audio file")
    y = librosa.util.normalize(y)
    return y

def extract_features(y, sr=16000):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_ddelta = librosa.feature.delta(mfcc, order=2)
    
    pitch, mag = librosa.piptrack(y=y, sr=sr)
    pitch = pitch[np.nonzero(pitch)]
    pitch_mean = np.mean(pitch) if pitch.size > 0 else 0

    energy = librosa.feature.rms(y=y)

    feature_vector = np.concatenate([
        np.mean(mfcc, axis=1),
        np.mean(mfcc_delta, axis=1),
        np.mean(mfcc_ddelta, axis=1),
        [pitch_mean],
        np.mean(energy, axis=1)
    ])
    return feature_vector

def save_spectrogram(y, sr=16000, output_path="spectrogram.png", n_mels=128, hop_length=512):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def build_label_dict(eval_dir_root):
    label_dict = {}
    for session_root, _, files in os.walk(eval_dir_root):
        for file in files:
            if file.endswith(".txt"):
                with open(os.path.join(session_root, file), "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5 and parts[4] != "xxx":
                            utt_id = parts[3]
                            emotion = parts[4]
                            label_dict[utt_id] = emotion
    return label_dict

def load_dataset(audio_dir, label_dict):
    X, y, speakers = [], [], []
    for root, _, files in os.walk(audio_dir):
        for file in files:
            if file.endswith(".wav") and not file.startswith("._"):
                path = os.path.join(root, file)
                utt_id = file.split(".")[0]
                label = label_dict.get(utt_id)
                if label:
                    try:
                        y_audio = load_audio(path)
                        features = extract_features(y_audio)
                        speaker_id = utt_id.split("_")[0]
                        X.append(features)
                        y.append(label)
                        speakers.append(speaker_id)
                    except Exception as e:
                        print(f"Error processing {file}: {e}")
    return np.array(X), np.array(y), np.array(speakers)
