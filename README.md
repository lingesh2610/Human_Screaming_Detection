🚨 Human Screaming Detection & Real-Time Emergency Alert System

An end-to-end, multi-layered personal security and emergency response system built with Python, PyTorch, Kivy, and Firebase Firestore. This project integrates real-time machine learning audio inference with an automated 2-minute family notification and police dispatch escalation protocol.

🌟 Key Features

Live AI Acoustic Detection: Uses a Convolutional Neural Network (CNN) trained on MFCC audio features extracted via Librosa to detect human screams from microphone streams in real time.

Dual-Mode Kivy Interface: Features a mobile-friendly desktop UI supporting both Victim Mode (personal monitoring & active emergency controls) and Police Dispatch Mode (real-time cloud-synced incident dashboard).

Automated 2-Minute Escalation: Triggers a 120-second family notification window upon detecting a distress signal. If unacknowledged by family contacts, the system automatically escalates the incident status to POLICE_DISPATCH.

Firebase Cloud Synchronization: Securely syncs GPS coordinates, timestamps, family contact details, and incident statuses in real time across devices using Firestore.

📸 Application Screenshots

1. Police Dispatch Mode (Cloud Synced Dashboard)

Displays active and past emergency incidents, confidence scores, GPS coordinates, timestamps, and live dispatch statuses.

[ INSERT SCREENSHOT OF POLICE MODE HERE ]
*(e.g., Image showing AI Scream Detected with 96.3% confidence and status updates)*


2. Victim Mode (Active Emergency & Countdown)

Shows the 2-minute family notification countdown and the safety confirmation button.


*(e.g., Image showing 🚨 CRITICAL ALERT ACTIVE! and time remaining)*


🏗️ System Architecture & Workflow

L1 Acoustic Detection: The microphone continuously records 3-second audio chunks; predict.py extracts MFCC features and classifies them via the PyTorch CNN model (screaming_model.pth).

Alert Triggering: Upon high-confidence detection or manual testing, an alert packet containing geolocation, phone number, and timestamp is generated.

Firestore Sync: The alert is pushed to Firebase Firestore (alerts collection).

Escalation Protocol:

Stage 1 (FAMILY_NOTIFYING): Family contact is alerted with a 2-minute countdown timer.

Stage 2 (CANCELED_BY_FAMILY): If the family member clicks "Confirm Safety & Cancel Dispatch", escalation is halted.

Stage 3 (POLICE_DISPATCH): If the timer expires, the incident automatically escalates to police authorities in the cloud dashboard.

🛠️ Installation & Setup

1. Clone the Repository

git clone https://github.com/YOUR_USERNAME/Human_screaming_detection.git
cd Human_screaming_detection


2. Create and Activate a Virtual Environment

python -m venv venv
venv\Scripts\Activate


3. Install Dependencies

pip install torch torchvision torchaudio librosa sounddevice soundfile kivy firebase-admin numpy


4. Configure Firebase

Create a free project on the Firebase Console.

Enable Cloud Firestore Database in test mode.

Go to Project Settings > Service accounts, generate a new private key for Python, rename the downloaded file to serviceAccountKey.json, and place it directly in the root project folder.

5. Run the Application

python main.py


📜 License

This project is open-source and developed for academic and personal safety innovation.
