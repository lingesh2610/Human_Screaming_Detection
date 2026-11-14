import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import soundfile as sf
from features import FeatureExtractor

class ScreamDataset(Dataset):
    """PyTorch Dataset for loading audio files and extracting MFCC features."""
    def __init__(self, data_dir, extractor):
        self.samples = []
        self.labels = []
        self.extractor = extractor

        # Load screaming samples (label 1)
        screaming_dir = os.path.join(data_dir, "screaming")
        if os.path.exists(screaming_dir):
            for fname in os.listdir(screaming_dir):
                if fname.endswith(".wav"):
                    self.samples.append(os.path.join(screaming_dir, fname))
                    self.labels.append(1)

        # Load non-screaming samples (label 0)
        not_screaming_dir = os.path.join(data_dir, "not_screaming")
        if os.path.exists(not_screaming_dir):
            for fname in os.listdir(not_screaming_dir):
                if fname.endswith(".wav"):
                    self.samples.append(os.path.join(not_screaming_dir, fname))
                    self.labels.append(0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path = self.samples[idx]
        label = self.labels[idx]

        try:
            audio, sr = sf.read(file_path)
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1) # Convert stereo to mono
            if sr != self.extractor.sample_rate:
                # Basic resample or assume correct sample rate
                pass

            features = self.extractor.extract_mfcc(audio)
            if features is None:
                features = np.zeros((self.extractor.n_mfcc, self.extractor.max_pad_len), dtype=np.float32)
            
            # Convert to PyTorch tensor and add channel dimension: (1, 40, 130)
            tensor_features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            tensor_label = torch.tensor(label, dtype=torch.long)

            return tensor_features, tensor_label

        except Exception as e:
            # Fallback for empty/corrupt files during training
            empty_features = torch.zeros((1, self.extractor.n_mfcc, self.extractor.max_pad_len), dtype=torch.float32)
            return empty_features, torch.tensor(label, dtype=torch.long)


class ScreamCNN(nn.Module):
    """Convolutional Neural Network for Audio Scream Classification."""
    def __init__(self):
        super(ScreamCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        # Calculated size after two 2x2 max poolings on (40, 130): (10, 32.5 -> 32) * 32 channels
        self.fc1 = nn.Linear(32 * 10 * 32, 64)
        self.fc2 = nn.Linear(64, 2) # Binary classification: [Not Screaming, Screaming]

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1) # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    print("Initializing PyTorch CNN Training Module...")
    extractor = FeatureExtractor()
    dataset_path = "dataset"
    
    if not os.path.exists(os.path.join(dataset_path, "screaming")) or not os.path.exists(os.path.join(dataset_path, "not_screaming")):
        print("[!] Dataset folders missing. Please ensure 'dataset/screaming' and 'dataset/not_screaming' exist with .wav files.")
    else:
        dataset = ScreamDataset(dataset_path, extractor)
        print(f"Loaded {len(dataset)} total audio samples for training.")
        
        if len(dataset) > 0:
            dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
            model = ScreamCNN()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            print("[*] Starting model training simulation (5 epochs)...")
            model.train()
            for epoch in range(5):
                running_loss = 0.0
                for inputs, labels in dataloader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                print(f"Epoch {epoch+1}/5 - Loss: {running_loss/len(dataloader):.4f}")

            os.makedirs("model", exist_ok=True)
            model_path = "model/screaming_model.pth"
            torch.save(model.state_dict(), model_path)
            print(f"[✓] Model successfully trained and saved to {model_path}!")
        else:
            print("[!] No .wav files found in dataset folders. Add a few sample wav files to test training.")