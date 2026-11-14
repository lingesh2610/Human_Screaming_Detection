"""
Human Screaming Detection & Alert System
Stage 2: Live Microphone Recording Module
"""

import numpy as np
import sounddevice as sd


class AudioRecorder:
    """Handles real-time microphone capture for acoustic analysis."""

    def __init__(self, sample_rate=16000, duration=3.0):
        self.sample_rate = sample_rate  # 16kHz is standard for lightweight audio ML
        self.duration = duration        # Length of audio chunk in seconds (3 seconds)

    def record_audio_chunk(self):
        """
        Records a short audio clip from the default system microphone.
        Returns a 1D numpy array of audio float samples.
        """
        print(f"[*] Recording {self.duration} seconds of audio...")
        try:
            # Record audio data synchronously from microphone
            audio_data = sd.rec(
                int(self.duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32'
            )
            sd.wait()  # Wait until the recording finishes
            
            # Flatten to 1D array
            audio_array = np.squeeze(audio_data)
            print("[*] Audio recording complete.")
            return audio_array
            
        except Exception as e:
            print(f"[!] Microphone recording error: {e}")
            return None


if __name__ == "__main__":
    # Quick standalone test for Stage 2
    print("Testing Microphone Recording (Speak or make a noise)...")
    recorder = AudioRecorder(duration=3.0)
    samples = recorder.record_audio_chunk()
    
    if samples is not None:
        print(f"Successfully recorded {len(samples)} audio samples!")
        print(f"Max amplitude: {np.max(np.abs(samples)):.4f}")
    else:
        print("Failed to record audio. Check microphone permissions.")