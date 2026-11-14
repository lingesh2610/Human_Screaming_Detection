"""
Human Screaming Detection & Alert System
Stage 3: MFCC Feature Extraction Module using Librosa
"""

import numpy as np
import librosa


class FeatureExtractor:
    """Extracts MFCC (Mel-Frequency Cepstral Coefficients) features from raw audio."""

    def __init__(self, sample_rate=16000, n_mfcc=40, max_pad_len=130):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc            # Number of MFCC features to extract
        self.max_pad_len = max_pad_len  # Fixed time frames for neural network input

    def extract_mfcc(self, audio_samples):
        """
        Takes a 1D numpy array of audio float samples and returns a 2D MFCC matrix.
        Pads or truncates the time dimension to ensure uniform shape.
        """
        try:
            # Extract MFCCs using Librosa
            mfccs = librosa.feature.mfcc(
                y=audio_samples,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc
            )

            # Pad or truncate to max_pad_len so all audio tensors match in shape
            if mfccs.shape[1] < self.max_pad_len:
                pad_width = self.max_pad_len - mfccs.shape[1]
                mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
            else:
                mfccs = mfccs[:, :self.max_pad_len]

            return mfccs

        except Exception as e:
            print(f"[!] Feature extraction error: {e}")
            return None


if __name__ == "__main__":
    # Quick standalone test for Stage 3
    print("Testing MFCC Feature Extraction...")
    extractor = FeatureExtractor()
    
    # Generate 3 seconds of dummy white noise to test extraction shape
    dummy_audio = np.random.randn(16000 * 3)
    features = extractor.extract_mfcc(dummy_audio)
    
    if features is not None:
        print(f"Successfully extracted MFCC features!")
        print(f"Feature tensor shape: {features.shape} (Expected: (40, 130))")
    else:
        print("Feature extraction failed.")