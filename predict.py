"""
Human Screaming Detection & Alert System
Stage 5: Real-time Live Inference & Prediction Module
"""

import os
import torch
import numpy as np
from detection import AudioRecorder
from features import FeatureExtractor
from train import ScreamCNN


class ScreamPredictor:
    """Loads the trained PyTorch CNN model and performs live audio classification."""

    def __init__(self, model_path="model/screaming_model.pth"):
        self.device = torch.device("cpu") # Use CPU for lightweight mobile/desktop inference
        self.extractor = FeatureExtractor()
        self.recorder = AudioRecorder(duration=3.0)
        
        # Initialize and load trained CNN weights
        self.model = ScreamCNN().to(self.device)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"[✓] Successfully loaded trained model from {model_path}")
        else:
            print(f"[!] Warning: Model weights not found at {model_path}. Predictions will be untrained.")

    def predict_live_audio(self):
        """Records a 3-second live audio clip and returns prediction probability."""
        audio_samples = self.recorder.record_audio_chunk()
        if audio_samples is None:
            return None, 0.0

        # Extract MFCC features
        features = self.extractor.extract_mfcc(audio_samples)
        if features is None:
            return None, 0.0

        # Prepare tensor for PyTorch model: batch_size=1, channels=1, shape=(40, 130)
        tensor_input = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor_input)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, dim=1)

        is_screaming = (predicted_class.item() == 1)
        conf_score = confidence.item()

        return is_screaming, conf_score


if __name__ == "__main__":
    print("Starting Live Scream Detection Test (Press Ctrl+C to stop)...")
    predictor = ScreamPredictor()

    try:
        while True:
            input("\nPress [Enter] to record and analyze a 3-second audio clip...")
            is_scream, confidence = predictor.predict_live_audio()
            
            if is_scream is not None:
                if is_scream and confidence > 0.80:
                    print(f"🚨 [ALERT] SCREAM DETECTED! (Confidence: {confidence * 100:.2f}%)")
                else:
                    print(f"✓ Normal audio environment. (Scream Confidence: {confidence * 100:.2f}%)")
    except KeyboardInterrupt:
        print("\nExiting live detection test.")