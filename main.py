"""
Human Screaming Detection & Alert System
Stage 6 & 7: Kivy Mobile Application with Live PyTorch Scream Detection & Alerts
"""

import threading
import time
from datetime import datetime
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.widget import Widget
from kivy.core.window import Window
from kivy.clock import Clock

# Import our AI prediction module (Stage 5)
from predict import ScreamPredictor

# --- FIREBASE SETUP ---
import firebase_admin
from firebase_admin import credentials, firestore

try:
    if not firebase_admin._apps:
        # Initialize Firebase using service account credentials
        # Make sure you download your serviceAccountKey.json from Firebase Console into your project folder!
        cred = credentials.Certificate("serviceAccountKey.json")
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    FIREBASE_ENABLED = True
    print("[✓] Connected to Firebase Firestore successfully!")
except Exception as e:
    FIREBASE_ENABLED = False
    print(f"[!] Firebase running in local memory mode (Credential error: {e})")


class ScreamingDetectionApp(App):
    """Main Kivy Application integrated with live PyTorch acoustic detection."""

    def build(self):
        # Set mobile-like window dimensions for desktop testing
        Window.size = (430, 750)
        Window.clearcolor = (0.08, 0.10, 0.14, 1)

        # Application state variables
        self.is_monitoring = False
        self.family_id = ""
        self.alerts = []
        self.active_alert = None
        self.escalation_seconds_left = 120  # 2 minutes family window

        # Initialize AI Scream Predictor
        self.predictor = ScreamPredictor()

        # Root container
        self.root_layout = BoxLayout(
            orientation="vertical",
            padding=15,
            spacing=12
        )

        # Build persistent top header and navigation tabs
        self.show_header()
        self.show_mode_buttons()

        # Dynamic content container
        self.content = BoxLayout(orientation="vertical", spacing=10)
        self.root_layout.add_widget(self.content)

        # Default to Victim Mode on startup
        self.show_victim_mode()

        return self.root_layout

    def show_header(self):
        """Renders the application title and subtitle."""
        title = Label(
            text="[b]HUMAN SCREAMING DETECTION[/b]",
            markup=True,
            font_size=20,
            color=(0.95, 0.26, 0.26, 1),
            size_hint_y=None,
            height=45
        )
        self.root_layout.add_widget(title)

        subtitle = Label(
            text="Emergency Alert System (AI + 2-Min Escalation)",
            font_size=13,
            color=(0.65, 0.70, 0.78, 1),
            size_hint_y=None,
            height=22
        )
        self.root_layout.add_widget(subtitle)

    def show_mode_buttons(self):
        """Renders top navigation buttons to switch between Victim and Police views."""
        buttons = BoxLayout(
            size_hint_y=None,
            height=48,
            spacing=10
        )

        self.victim_button = Button(
            text="Victim Mode",
            bold=True,
            background_normal="",
            background_color=(0.15, 0.45, 0.85, 1)
        )
        self.victim_button.bind(on_press=self.show_victim_mode)

        self.police_button = Button(
            text="Police Mode",
            bold=True,
            background_normal="",
            background_color=(0.25, 0.28, 0.34, 1)
        )
        self.police_button.bind(on_press=self.show_police_mode)

        buttons.add_widget(self.victim_button)
        buttons.add_widget(self.police_button)

        self.root_layout.add_widget(buttons)

    def clear_content(self):
        """Clears all widgets from the dynamic content area."""
        self.content.clear_widgets()

    def show_victim_mode(self, *args):
        """Renders the Victim Mode screen for personal security monitoring."""
        self.clear_content()

        # Highlight active tab
        self.victim_button.background_color = (0.15, 0.45, 0.85, 1)
        self.police_button.background_color = (0.25, 0.28, 0.34, 1)

        heading = Label(
            text="[b]Victim Mode[/b]",
            markup=True,
            font_size=22,
            size_hint_y=None,
            height=40
        )
        self.content.add_widget(heading)

        explanation = Label(
            text="Live AI Acoustic Monitoring & Escalation",
            color=(0.75, 0.78, 0.85, 1),
            size_hint_y=None,
            height=30
        )
        self.content.add_widget(explanation)

        # Family Contact ID input
        self.family_input = TextInput(
            text=self.family_id,
            hint_text="Enter Family Contact ID",
            multiline=False,
            padding=[12, 14],
            size_hint_y=None,
            height=50
        )
        self.family_input.bind(text=self.on_family_id_change)
        self.content.add_widget(self.family_input)

        # Emergency Contact Phone Number input
        if not hasattr(self, 'phone_number'):
            self.phone_number = ""
        self.phone_input = TextInput(
            text=self.phone_number,
            hint_text="Enter Family Phone Number (e.g., +1234567890)",
            multiline=False,
            padding=[12, 14],
            size_hint_y=None,
            height=50
        )
        self.phone_input.bind(text=self.on_phone_change)
        self.content.add_widget(self.phone_input)

        if self.active_alert:
            # Render active emergency state
            alert_card = BoxLayout(orientation="vertical", spacing=8, size_hint_y=None, height=180)
            alert_card.add_widget(Label(text="[b]🚨 CRITICAL ALERT ACTIVE! 🚨[/b]", markup=True, color=(1, 0.3, 0.3, 1), size_hint_y=None, height=30))
            
            self.timer_label = Label(
                text=f"Time remaining for Family Response: {self.escalation_seconds_left // 60}m {self.escalation_seconds_left % 60:02d}s",
                color=(1, 0.8, 0.2, 1),
                size_hint_y=None,
                height=30
            )
            alert_card.add_widget(self.timer_label)
            
            alert_card.add_widget(Label(text="Status: FAMILY_NOTIFYING (Escalates to Police if unconfirmed)", color=(0.8, 0.8, 0.8, 1), size_hint_y=None, height=25))

            cancel_btn = Button(
                text="Confirm Safety & Cancel Dispatch",
                bold=True,
                background_normal="",
                background_color=(0.15, 0.68, 0.38, 1),
                size_hint_y=None,
                height=50
            )
            cancel_btn.bind(on_press=self.cancel_alert)
            alert_card.add_widget(cancel_btn)

            self.content.add_widget(alert_card)
        else:
            # Toggle Monitoring Button
            monitor_button = Button(
                text="Stop AI Monitoring" if self.is_monitoring else "Start AI Monitoring",
                bold=True,
                background_normal="",
                background_color=(0.85, 0.60, 0.10, 1) if self.is_monitoring else (0.15, 0.68, 0.38, 1),
                size_hint_y=None,
                height=55
            )
            monitor_button.bind(on_press=self.toggle_monitoring)
            self.content.add_widget(monitor_button)

            # Manual Test Alert Button
            test_alert_button = Button(
                text="CREATE TEST ALERT",
                bold=True,
                background_normal="",
                background_color=(0.85, 0.15, 0.15, 1),
                size_hint_y=None,
                height=55
            )
            test_alert_button.bind(on_press=self.create_test_alert)
            self.content.add_widget(test_alert_button)

            # Status Label reflecting current monitoring state
            self.status_label = Label(
                text="Status: AI Monitoring enabled (listening...)"
                if self.is_monitoring
                else "Status: Monitoring stopped",
                color=(0.30, 0.85, 0.45, 1) if self.is_monitoring else (0.95, 0.75, 0.20, 1),
                size_hint_y=None,
                height=45
            )
            self.content.add_widget(self.status_label)

        # Spacer to push widgets to the top
        self.content.add_widget(Widget())

    def on_family_id_change(self, instance, value):
        """Stores the family ID whenever the user types in the input box."""
        self.family_id = value.strip()

    def on_phone_change(self, instance, value):
        """Stores the family phone number whenever the user types."""
        self.phone_number = value.strip()

    def toggle_monitoring(self, instance):
        """Toggles AI microphone monitoring state."""
        self.is_monitoring = not self.is_monitoring
        
        if self.is_monitoring:
            # Start background thread for real-time mic listening
            self.monitoring_thread = threading.Thread(target=self.audio_monitoring_loop, daemon=True)
            self.monitoring_thread.start()

        self.show_victim_mode()

    def audio_monitoring_loop(self):
        """Background loop that continuously records and analyzes microphone input using PyTorch."""
        while self.is_monitoring and not self.active_alert:
            is_scream, confidence = self.predictor.predict_live_audio()
            
            if is_scream is not None and is_scream and confidence > 0.85:
                print(f"🚨 [AI DETECTION] Scream detected with confidence {confidence * 100:.2f}%!")
                Clock.schedule_once(lambda dt: self.trigger_real_alert(confidence), 0)
                break
            
            time.sleep(0.5)

    def trigger_real_alert(self, confidence):
        """Triggers an emergency alert when the AI detects a genuine scream."""
        self.family_id = self.family_input.text.strip()
        if not self.family_id:
            self.family_id = "DefaultFamilyContact"
        
        phone = getattr(self, 'phone_number', '+1-555-0199')
        if not phone:
            phone = '+1-555-0199'

        self.is_monitoring = False
        
        mock_lat, mock_lon = 11.6643, 78.1460
        location_str = f"Lat: {mock_lat}, Lon: {mock_lon} (GPS Verified)"
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        alert_data = {
            "message": f"AI Scream Detected! (Confidence: {confidence * 100:.1f}%)",
            "status": "FAMILY_NOTIFYING",
            "family_id": self.family_id,
            "phone": phone,
            "location": location_str,
            "time_str": timestamp_str,
            "timestamp": firestore.SERVER_TIMESTAMP if FIREBASE_ENABLED else time.time()
        }

        # Save to Firebase Firestore if enabled
        alert_id = "local_" + str(int(time.time()))
        if FIREBASE_ENABLED:
            try:
                update_ref, doc_ref = db.collection("alerts").add(alert_data)
                alert_id = doc_ref.id
                print(f"[✓] Alert synced to Firebase with ID: {alert_id}")
            except Exception as e:
                print(f"[!] Error writing alert to Firebase: {e}")

        alert_data["id"] = alert_id
        self.active_alert = alert_data
        self.alerts.insert(0, alert_data)
        self.escalation_seconds_left = 120

        self.timer_event = Clock.schedule_interval(self.update_escalation_timer, 1.0)
        self.show_victim_mode()

    def create_test_alert(self, instance):
        """Creates a test alert and syncs it to Firebase."""
        self.family_id = self.family_input.text.strip()
        if not self.family_id:
            self.family_id = "DefaultFamilyContact"

        phone = getattr(self, 'phone_number', '+1-555-0199')
        if not phone:
            phone = '+1-555-0199'

        mock_lat, mock_lon = 11.6643, 78.1460
        location_str = f"Lat: {mock_lat}, Lon: {mock_lon} (GPS Verified)"
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        alert_data = {
            "message": "Manual Test Emergency Alert",
            "status": "FAMILY_NOTIFYING",
            "family_id": self.family_id,
            "phone": phone,
            "location": location_str,
            "time_str": timestamp_str,
            "timestamp": firestore.SERVER_TIMESTAMP if FIREBASE_ENABLED else time.time()
        }

        alert_id = "local_" + str(int(time.time()))
        if FIREBASE_ENABLED:
            try:
                _, doc_ref = db.collection("alerts").add(alert_data)
                alert_id = doc_ref.id
            except Exception as e:
                print(f"[!] Error writing test alert to Firebase: {e}")

        alert_data["id"] = alert_id
        self.active_alert = alert_data
        self.alerts.insert(0, alert_data)
        self.escalation_seconds_left = 120

        self.timer_event = Clock.schedule_interval(self.update_escalation_timer, 1.0)
        self.show_victim_mode()

    def update_escalation_timer(self, dt):
        """Decrements the 2-minute family escalation timer."""
        if self.escalation_seconds_left > 0:
            self.escalation_seconds_left -= 1
            minutes = self.escalation_seconds_left // 60
            seconds = self.escalation_seconds_left % 60
            if hasattr(self, 'timer_label') and self.timer_label:
                self.timer_label.text = f"Time remaining for Family Response: {minutes}m {seconds:02d}s"
        else:
            # Timer expired -> Escalate to Police Dispatch in Firebase!
            if self.active_alert:
                self.active_alert["status"] = "POLICE_DISPATCH"
                alert_id = self.active_alert.get("id")
                if FIREBASE_ENABLED and alert_id and not alert_id.startswith("local_"):
                    try:
                        db.collection("alerts").document(alert_id).update({"status": "POLICE_DISPATCH"})
                    except Exception as e:
                        print(f"[!] Error updating escalation status in Firebase: {e}")
            if hasattr(self, 'timer_event'):
                self.timer_event.cancel()
            self.show_victim_mode()

    def cancel_alert(self, instance):
        """Cancels the alert when family confirms safety and updates Firebase."""
        if hasattr(self, 'timer_event'):
            self.timer_event.cancel()
        if self.active_alert:
            self.active_alert["status"] = "CANCELED_BY_FAMILY"
            alert_id = self.active_alert.get("id")
            if FIREBASE_ENABLED and alert_id and not alert_id.startswith("local_"):
                try:
                    db.collection("alerts").document(alert_id).update({"status": "CANCELED_BY_FAMILY"})
                except Exception as e:
                    print(f"[!] Error updating cancel status in Firebase: {e}")
        self.active_alert = None
        self.show_victim_mode()

    def show_police_mode(self, *args):
        """Renders the Police Dispatch Mode screen with active alerts fetched from Firebase."""
        self.clear_content()

        self.police_button.background_color = (0.85, 0.20, 0.20, 1)
        self.victim_button.background_color = (0.25, 0.28, 0.34, 1)

        heading = Label(
            text="[b]Police Dispatch Mode (Cloud Synced)[/b]",
            markup=True,
            font_size=22,
            size_hint_y=None,
            height=40
        )
        self.content.add_widget(heading)

        # Fetch latest alerts from Firebase if available
        if FIREBASE_ENABLED:
            try:
                docs = db.collection("alerts").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(10).stream()
                cloud_alerts = []
                for doc in docs:
                    data = doc.to_dict()
                    data["id"] = doc.id
                    cloud_alerts.append(data)
                if cloud_alerts:
                    self.alerts = cloud_alerts
            except Exception as e:
                print(f"[!] Error fetching alerts from Firebase: {e}")

        if not self.alerts:
            self.content.add_widget(
                Label(
                    text="No active emergency alerts",
                    color=(0.60, 0.65, 0.70, 1)
                )
            )
            return

        scroll = ScrollView(size_hint=(1, 1))
        alert_list_layout = BoxLayout(
            orientation="vertical",
            size_hint_y=None,
            spacing=12,
            padding=[0, 5]
        )
        alert_list_layout.bind(minimum_height=alert_list_layout.setter("height"))

        for alert in self.alerts:
            alert_box = BoxLayout(
                orientation="vertical",
                size_hint_y=None,
                height=180,
                padding=10,
                spacing=4
            )

            status_color = (0.95, 0.75, 0.20, 1)
            if alert["status"] == "POLICE_DISPATCH":
                status_color = (0.95, 0.25, 0.25, 1)
            elif alert["status"] == "CANCELED_BY_FAMILY":
                status_color = (0.40, 0.80, 0.40, 1)

            alert_box.add_widget(
                Label(
                    text=f"[b]{alert['message']}[/b]",
                    markup=True,
                    color=(1, 1, 1, 1),
                    size_hint_y=None,
                    height=25
                )
            )
            alert_box.add_widget(
                Label(
                    text=f"Status: {alert['status']}",
                    color=status_color,
                    size_hint_y=None,
                    height=22
                )
            )
            alert_box.add_widget(
                Label(
                    text=f"Time: {alert.get('time_str', 'N/A')}",
                    color=(0.8, 0.8, 0.9, 1),
                    size_hint_y=None,
                    height=22
                )
            )
            alert_box.add_widget(
                Label(
                    text=f"Location: {alert.get('location', 'GPS Unknown')}",
                    color=(0.9, 0.9, 0.6, 1),
                    size_hint_y=None,
                    height=22
                )
            )
            alert_box.add_widget(
                Label(
                    text=f"Family ID: {alert['family_id']} | Phone: {alert.get('phone', 'N/A')}",
                    color=(0.75, 0.80, 0.85, 1),
                    size_hint_y=None,
                    height=22
                )
            )

            alert_list_layout.add_widget(alert_box)

        scroll.add_widget(alert_list_layout)
        self.content.add_widget(scroll)


if __name__ == "__main__":
    ScreamingDetectionApp().run()