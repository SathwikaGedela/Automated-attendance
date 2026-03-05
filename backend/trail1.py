from flask import Flask, request, jsonify
from flask_cors import CORS
import face_recognition
import numpy as np
import os
import csv
from datetime import datetime, timedelta, date
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = Flask(__name__)
CORS(app)  

# Dictionary mapping names to image paths
image_files = {
    "harika": "photos/harika.jpg",
    "lahari": "photos/lahari.jpg",
    "sathwika": "photos/sathwika.jpg"
}

# Load known face encodings
known_face_encodings = []
known_face_names = []

for name, file_path in image_files.items():
    if os.path.exists(file_path):
        image = face_recognition.load_image_file(file_path)
        face_locations = face_recognition.face_locations(image, model="hog")  # Avoids dlib build issues
        if face_locations:
            encoding = face_recognition.face_encodings(image, face_locations)[0]
            known_face_encodings.append(encoding)
            known_face_names.append(name)
            print(f"Loaded face for {name}")
        else:
            print(f"No face detected in {file_path}")

attendance_records = []

# List of holidays
holidays = [
    {"name": "Republic Day", "date": "2025-01-26"},
    {"name": "Independence Day", "date": "2025-08-15"},
    {"name": "Gandhi Jayanti", "date": "2025-10-02"},
    {"name": "Holi", "date": "2025-03-06"},
    {"name": "Diwali", "date": "2025-10-23"},
    {"name": "Christmas", "date": "2025-12-25"},
    {"name": "New Year's Day", "date": "2025-01-01"}
]

# Configuration for attendance tracking and notifications
attendance_config = {
    "expected_arrival_time": "09:30:00",  # Expected time of arrival
    "absence_threshold_days": 2,  # Number of consecutive days absent to trigger alert
    "irregular_pattern_threshold": 3,  # Number of late arrivals in a week to trigger alert
    "enable_email_notifications": True,  # Set to True to enable email notifications
    "notification_email": {
        "sender": "gowthamsai630@gmail.com",
        "password": "aumszxzvfyaxrojd",  # Secure this in production!
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "recipients": ["sathwikagedela@gmail.com"]
    },
    "work_days": [0, 1, 2, 3, 4]  # Monday=0, Tuesday=1, ..., Friday=4
}

# Store historical attendance data
attendance_history = {}  # Format: {name: {date: time}}
late_arrival_counts = {}  # Format: {name: count_of_late_arrivals_this_week}
absence_streak = {}  # Format: {name: consecutive_days_absent}

def is_holiday(check_date):
    """Check if a given date is a holiday"""
    date_str = check_date.strftime("%Y-%m-%d")
    return any(holiday["date"] == date_str for holiday in holidays)

def is_weekend(check_date):
    """Check if a given date is a weekend"""
    return check_date.weekday() not in attendance_config["work_days"]

def send_notification_email(subject, message):
    """Send notification email"""
    if not attendance_config["enable_email_notifications"]:
        print(f"Email notification would be sent: {subject}")
        return
        
    try:
        config = attendance_config["notification_email"]
        msg = MIMEMultipart()
        msg['From'] = config["sender"]
        msg['To'] = ", ".join(config["recipients"])
        msg['Subject'] = subject
        
        msg.attach(MIMEText(message, 'plain'))
        
        server = smtplib.SMTP(config["smtp_server"], config["smtp_port"])
        server.starttls()
        server.login(config["sender"], config["password"])
        server.send_message(msg)
        server.quit()
        print(f"Email notification sent: {subject}")
    except Exception as e:
        print(f"Failed to send email notification: {e}")

def check_attendance_patterns():
    """Check for absence and irregular attendance patterns"""
    today = date.today()
    notifications = []
    
    # Initialize absence streak for any new person
    for name in known_face_names:
        if name not in absence_streak:
            absence_streak[name] = 0
    
    # Check who is present today
    present_today = set()
    for record in attendance_records:
        if record.get("date") == today.strftime("%Y-%m-%d"):
            present_today.add(record["name"])
    
    # Update absence streaks and check for absences
    for name in known_face_names:
        if name in present_today:
            # Reset absence streak
            absence_streak[name] = 0
        else:
            # Skip weekends and holidays
            if is_weekend(today) or is_holiday(today):
                continue
                
            # Increment absence streak
            absence_streak[name] = absence_streak.get(name, 0) + 1
            
            # Check if absence threshold is reached
            if absence_streak[name] >= attendance_config["absence_threshold_days"]:
                msg = f"{name} has been absent for {absence_streak[name]} consecutive working days."
                notifications.append({"type": "absence", "message": msg})
                send_notification_email(f"Absence Alert: {name}", msg)
    
    # Check for irregular patterns (late arrivals)
    for record in attendance_records:
        if record.get("date") == today.strftime("%Y-%m-%d"):
            expected_time = datetime.strptime(attendance_config["expected_arrival_time"], "%H:%M:%S").time()
            arrival_time = datetime.strptime(record["time"], "%H:%M:%S").time()
            
            if arrival_time > expected_time:
                # Count late arrivals this week
                name = record["name"]
                late_arrival_counts[name] = late_arrival_counts.get(name, 0) + 1
                
                # Check if irregular pattern threshold is reached
                if late_arrival_counts[name] >= attendance_config["irregular_pattern_threshold"]:
                    msg = f"{name} has arrived late {late_arrival_counts[name]} times this week."
                    notifications.append({"type": "irregular", "message": msg})
                    send_notification_email(f"Irregular Attendance Alert: {name}", msg)
    
    # Once a week (Monday), reset the late arrival counters
    if today.weekday() == 0:  # Monday
        late_arrival_counts.clear()
        
    return notifications

@app.route('/recognize', methods=['POST'])
def recognize_faces():
    global attendance_records
    file = request.files['image']
    file_path = "captured.jpg"
    file.save(file_path)

    # Load image and detect faces
    img = face_recognition.load_image_file(file_path)
    face_locations = face_recognition.face_locations(img, model="hog")  
    face_encodings = face_recognition.face_encodings(img, face_locations)

    recognized_attendance = []
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        name = "Unknown"

        if True in matches:
            best_match_index = matches.index(True)
            name = known_face_names[best_match_index]

        recognized_attendance.append({"name": name, "time": current_time, "date": current_date})

        if name != "Unknown":
            attendance_record = {"name": name, "time": current_time, "date": current_date}
            attendance_records.append(attendance_record)
            
            # Update attendance history
            if name not in attendance_history:
                attendance_history[name] = {}
            attendance_history[name][current_date] = current_time

    # Save attendance to CSV (append instead of overwrite)
    csv_filename = f"attendance_{current_date}.csv"
    
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Name", "Time", "Date"])  
        for record in recognized_attendance:
            if record["name"] != "Unknown":
                writer.writerow([record["name"], record["time"], record["date"]])

    # Check attendance patterns and generate notifications
    notifications = check_attendance_patterns()

    return jsonify({
        "attendance": recognized_attendance,
        "notifications": notifications
    })

@app.route('/attendance', methods=['GET'])
def get_attendance():
    return jsonify({"attendance": attendance_records})

@app.route('/download_csv', methods=['GET'])
def download_csv():
    current_date = datetime.now().strftime("%Y-%m-%d")
    csv_filename = f"attendance_{current_date}.csv"
    return jsonify({"csv_file": csv_filename})

@app.route('/holidays', methods=['GET'])
def get_holidays():
    return jsonify({"holidays": holidays})

@app.route('/notifications', methods=['GET'])
def get_notifications():
    """Get current notifications about absence and irregular patterns"""
    notifications = check_attendance_patterns()
    return jsonify({"notifications": notifications})

@app.route('/attendance_history/<name>', methods=['GET'])
def get_attendance_history(name):
    """Get attendance history for a specific person"""
    if name in attendance_history:
        return jsonify({"history": attendance_history[name]})
    else:
        return jsonify({"error": f"No attendance records found for {name}"}), 404

@app.route('/configure_notifications', methods=['POST'])
def configure_notifications():
    """Update notification configuration"""
    data = request.json
    
    # Update configuration with received values
    for key, value in data.items():
        if key in attendance_config:
            attendance_config[key] = value
            
    return jsonify({"message": "Notification settings updated", "config": attendance_config})

if __name__ == "__main__":
    app.run(debug=True)