# -*- coding: utf-8 -*-
"""fusion_bids_final.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1WPa1yqzsH_LeiWcWsdSJqgMBC6rW-C8G
"""

!pip install transformers --quiet

from google.colab import drive
drive.mount('/content/drive/')

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ================================
# 🔤 Load Data with Transcripts + Personality Labels
# ================================
def load_iemocap_fusion(root_dir, target_labels=None):
    import os
    dataset = []
    label_map = {
        'hap': [0.8, 0.5, 0.9, 0.7, 0.2],
        'sad': [0.4, 0.5, 0.2, 0.6, 0.9],
        'ang': [0.3, 0.4, 0.4, 0.3, 0.8],
        'neu': [0.5, 0.8, 0.6, 0.6, 0.3]
    }

    sessions = [os.path.join(root_dir, s) for s in os.listdir(root_dir) if s.startswith("Session")]

    for session in sessions:
        wav_root = os.path.join(session, "sentences/wav")
        trans_root = os.path.join(session, "dialog/transcriptions")
        label_folder = os.path.join(session, "dialog/EmoEvaluation")

        if not os.path.exists(label_folder): continue

        for emo_file in os.listdir(label_folder):
            if emo_file.endswith(".txt"):
                emo_path = os.path.join(label_folder, emo_file)

                with open(emo_path, "r") as f:
                    lines = f.readlines()

                for line in lines:
                    if line.startswith('['):
                        parts = line.strip().split('\t')
                        if len(parts) >= 3:
                            filename = parts[1].strip()
                            emotion = parts[2].strip()

                            if target_labels is None or emotion in target_labels:
                                transcript = ""
                                transcript_file = os.path.join(trans_root, emo_file)
                                if os.path.exists(transcript_file):
                                    with open(transcript_file, "r") as tf:
                                        for t_line in tf:
                                            if filename in t_line:
                                                transcript = t_line.split(":")[-1].strip()
                                                break

                                # ✅ Search recursively for .wav file
                                audio_path = None
                                for root, _, files in os.walk(wav_root):
                                    if filename + ".wav" in files:
                                        audio_path = os.path.join(root, filename + ".wav")
                                        break

                                if audio_path and transcript:
                                    dataset.append((audio_path, transcript, label_map[emotion]))

    print(f"✅ Loaded {len(dataset)} audio-text-label samples")
    return dataset

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()

class FusionDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        audio_path, text, label = self.data[idx]
        y, sr = librosa.load(audio_path, sr=16000)
        y = librosa.util.normalize(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        audio_tensor = torch.tensor(mfcc_mean, dtype=torch.float32)

        with torch.no_grad():
            tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
            output = bert_model(**tokens)
            text_embedding = output.last_hidden_state[0, 0, :]
        text_tensor = text_embedding.float()
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return audio_tensor, text_tensor, label_tensor

# ================================
# 🧠 Fusion Model
# ================================
class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()
        self.audio_branch = nn.Sequential(nn.Linear(40, 128), nn.ReLU())
        self.text_branch = nn.Sequential(nn.Linear(768, 128), nn.ReLU())
        self.fusion = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 5)
        )
    def forward(self, audio_feat, text_feat):
        a = self.audio_branch(audio_feat)
        t = self.text_branch(text_feat)
        x = torch.cat([a, t], dim=1)
        return self.fusion(x)

# 🚂 Train Function
# ================================
def train(model, dataloader, epochs=10):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for audio, text, target in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            audio, text, target = audio.to(device), text.to(device), target.to(device)
            output = model(audio, text)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {running_loss:.4f}")

def plot_radar(personality_scores, labels=["O", "C", "E", "A", "N"], show=True):
    values = personality_scores.tolist()
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(values))
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.3)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    plt.title("Predicted Big Five Personality Traits")
    if show:
        plt.show()
    return fig

import torch

def save_model(model, path="fusion_model.pth"):
    torch.save(model.state_dict(), path)
    print(f"✅ Model saved to {path}")

def load_model(model, path="fusion_model.pth", device="cpu"):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"✅ Model loaded from {path}")
    return model

IEMOCAP_PATH = "/content/drive/MyDrive/SpeechMajorProjectFolder"
TARGET_EMOTIONS = ['hap', 'ang', 'sad', 'neu']

# Load dataset
dataset = load_iemocap_fusion(IEMOCAP_PATH, target_labels=TARGET_EMOTIONS)
train_data, test_data = train_test_split(dataset, test_size=0.2)

!pip install fpdf
from report_generator import generate_pdf, plot_radar

if __name__ == "__main__":


    # Create datasets
    train_dataset = FusionDataset(train_data)
    test_dataset = FusionDataset(test_data)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Model
    model = FusionModel()
    train(model, train_loader, epochs=10)

    torch.save(model.state_dict(), "fusion_model.pth")
    print("Model saved to fusion_model.pth")

# Predict on one test sample
# Set model to evaluation mode
model.eval()

# Move model to device (CPU in your case)
model.to(device)

# Get a sample
sample_audio, sample_text, _ = test_dataset[1]

# Run prediction on the correct device
with torch.no_grad():
    pred = model(
        sample_audio.unsqueeze(0).to(device),
        sample_text.unsqueeze(0).to(device)
    ).squeeze().cpu()

# Visualize output
plot_radar(pred)

# predrep = model()
fig = plot_radar(pred, show=False)
generate_pdf(pred, fig, output_path="report.pdf")

from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def evaluate_fusion_model(test_dataset, model, save_reports=False, report_dir="fusion_reports"):
    model.eval()
    os.makedirs(report_dir, exist_ok=True)

    all_preds, all_targets, ids = [], [], []

    for idx in tqdm(range(len(test_dataset))):
        audio_feat, text_feat, target = test_dataset[idx]
        with torch.no_grad():
            audio_feat = audio_feat.unsqueeze(0).to(device)
            text_feat = text_feat.unsqueeze(0).to(device)
            pred = model(audio_feat, text_feat).squeeze().cpu().numpy()

        target = target.numpy()
        all_preds.append(pred)
        all_targets.append(target)
        ids.append(f"sample_{idx}")

        if save_reports:
            fig = plot_radar(torch.tensor(pred), show=False)
            generate_pdf(pred, fig, output_path=os.path.join(report_dir, f"{ids[-1]}_report.pdf"))

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Compute metrics
    mae_total = mean_absolute_error(all_targets, all_preds)
    mae_per_trait = mean_absolute_error(all_targets, all_preds, multioutput='raw_values')
    r2_scores = r2_score(all_targets, all_preds, multioutput='raw_values')
    r2_avg = r2_score(all_targets, all_preds)

    trait_names = ["O", "C", "E", "A", "N"]

    print(f"Overall MAE: {mae_total:.4f}")
    print(f"R² Score (Avg): {r2_avg:.4f}")
    for i, t in enumerate(trait_names):
        print(f"   - {t}: MAE={mae_per_trait[i]:.4f}, R²={r2_scores[i]:.4f}")

    # Save to CSV
    df = pd.DataFrame({
        "sample_id": ids,
        **{f"{t}_true": all_targets[:, i] for i, t in enumerate(trait_names)},
        **{f"{t}_pred": all_preds[:, i] for i, t in enumerate(trait_names)},
    })
    df.to_csv(os.path.join(report_dir, "fusion_evaluation_summary.csv"), index=False)
    print(f"Summary saved to {report_dir}/fusion_evaluation_summary.csv")

evaluate_fusion_model(test_dataset=test_dataset, model=model, save_reports=True)

df = pd.read_csv("fusion_reports/fusion_evaluation_summary.csv")
traits = ["O", "C", "E", "A", "N"]

# Threshold for closeness
threshold = 0.30

# Trait-wise accuracy
trait_correct = {t: 0 for t in traits}
sample_correct = 0

for i in range(len(df)):
    correct_traits = 0
    for t in traits:
        true_val = df.loc[i, f"{t}_true"]
        pred_val = df.loc[i, f"{t}_pred"]
        if abs(true_val - pred_val) <= threshold:
            trait_correct[t] += 1
            correct_traits += 1

    if correct_traits == len(traits):
        sample_correct += 1

# Trait-wise accuracy
total = len(df)
print(f"\n Accuracy with threshold ±{threshold:.2f}")
for t in traits:
    acc = trait_correct[t] / total * 100
    print(f"   - {t}: {acc:.2f}%")

# Overall sample-level accuracy
sample_acc = sample_correct / total * 100
print(f"\n All-trait (sample-level) accuracy: {sample_acc:.2f}%")

import shutil

shutil.make_archive("fusion_reports", 'zip', "fusion_reports")

