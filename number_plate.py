import cv2
from ultralytics import YOLO
import numpy as np
import math
import re
import os
from datetime import datetime
import sys
import easyocr
import torch

import firebase_admin
from firebase_admin import credentials, db

# ----- NEW: MQTT -----
import paho.mqtt.client as mqtt
# MQTT configuration
MQTT_BROKER = "192.168.0.10"  # <-- change to your broker IP/hostname
MQTT_PORT = 1883
MQTT_USER = None              # e.g., "username" or None
MQTT_PASS = None              # e.g., "password" or None
MQTT_TOPIC = "ranger/control/servo"
MQTT_QOS = 1
SERVO_TRIGGER_COOLDOWN_SEC = 2.0  # avoid spamming

# ---------- Config ----------
FIREBASE_CREDENTIALS_PATH = r"ranger_database.json"
FIREBASE_DB_URL = "https://ranger-e0a95-default-rtdb.firebaseio.com/"
rtsp_url = "http://192.168.0.242:8080/stream.mjpg"  # <-- set your RTSP stream here
yolo_weights = "best.pt"                # your license-plate detector weights

# Workaround for some OpenMP/KMP conflicts on Windows/conda
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---------- CUDA / device ----------
try:
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        pass
except Exception:
    pass

USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if USE_CUDA else "cpu"
print(f"[INFO] Torch CUDA available: {USE_CUDA} -> using {DEVICE}")

# ---------- Firebase ----------
def init_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
        firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})
    return db.reference("/license_events"), db.reference("/license_events_unique")

fb_events_ref, fb_unique_ref = init_firebase()
sent_to_firebase = set()  # de-dup in-process

def load_existing_plates_once():
    try:
        unique_snapshot = fb_unique_ref.get()
        if isinstance(unique_snapshot, dict):
            for _, payload in unique_snapshot.items():
                plate = None
                if isinstance(payload, dict):
                    plate = payload.get("number_plate")
                elif isinstance(payload, str):
                    plate = payload
                if plate:
                    sent_to_firebase.add(plate)
        else:
            events_snapshot = fb_events_ref.get()
            if isinstance(events_snapshot, dict):
                for _, payload in events_snapshot.items():
                    if isinstance(payload, dict):
                        plate = payload.get("number_plate")
                        if plate:
                            sent_to_firebase.add(plate)
    except Exception as e:
        print(f"[WARN] Firebase preload failed: {e}")

load_existing_plates_once()

def push_plate_to_firebase_once(plate_text, detected_time_iso):
    if not plate_text:
        return
    if plate_text in sent_to_firebase:
        return

    safe_key = (
        plate_text.replace(".", "_")
                  .replace("$", "_")
                  .replace("[", "_")
                  .replace("]", "_")
                  .replace("#", "_")
                  .replace("/", "_")
    )
    plate_ref = fb_unique_ref.child(safe_key)
    try:
        existing = plate_ref.get()
        if existing:
            sent_to_firebase.add(plate_text)
            return

        plate_ref.set({
            "number_plate": plate_text,
            "first_detected_at": detected_time_iso
        })
        fb_events_ref.child(safe_key).set({
            "number_plate": plate_text,
            "detected_at": detected_time_iso
        })
        sent_to_firebase.add(plate_text)
    except Exception as e:
        print(f"[WARN] Firebase write failed: {e}")
        # Avoid spamming retries if Firebase is failing; still mark to prevent repeated writes
        sent_to_firebase.add(plate_text)
        return

# ---------- EasyOCR ----------
def init_easyocr_reader():
    try:
        reader_gpu = easyocr.Reader(['en'], gpu=USE_CUDA)
        print(f"[INFO] EasyOCR initialized with gpu={USE_CUDA}.")
        return reader_gpu
    except Exception as e:
        print(f"[WARN] EasyOCR GPU init failed ({e}), falling back to CPU...")
        reader_cpu = easyocr.Reader(['en'], gpu=False)
        print("[INFO] EasyOCR initialized with gpu=False.")
        return reader_cpu

reader = init_easyocr_reader()

def easyocr_ocr(frame, x1, y1, x2, y2):
    H, W = frame.shape[:2]
    pad = 5
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(W, x2 + pad); y2 = min(H, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return ""

    crop = frame[y1:y2, x1:x2]
    h, w = crop.shape[:2]
    if w < 60 or h < 20:
        return ""

    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        bin_img = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5
        )
    except Exception:
        bin_img = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    def run_easyocr(img):
        try:
            return reader.readtext(
                img, detail=1, paragraph=False, batch_size=1,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- "
            )
        except Exception:
            return []

    results = run_easyocr(bin_img)
    if not results:
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = run_easyocr(crop_rgb)

    best_text, best_conf = "", 0.0
    for item in results:
        if len(item) >= 3:
            _, text, conf = item
            try:
                conf = float(conf)
            except Exception:
                continue
            cleaned = re.sub(r"[^A-Z0-9-]", "", text.upper())
            if len(cleaned) >= 5 and conf > best_conf:
                best_text, best_conf = cleaned, conf

    if best_conf < 0.45:
        return ""

    best_text = best_text.replace("O", "0")
    return best_text

# ---------- YOLO ----------
model = YOLO(yolo_weights)
try:
    if USE_CUDA:
        model.to('cuda')
        print("[INFO] YOLO model moved to CUDA.")
    else:
        print("[INFO] YOLO model running on CPU.")
except Exception as e:
    print(f"[WARN] Could not move YOLO model to CUDA explicitly: {e}. It may still use GPU via device='cuda' during predict.")

# Ultralytics predict kwargs
PREDICT_KW = {"conf": 0.45, "verbose": False}
PREDICT_KW["device"] = 0 if USE_CUDA else "cpu"
print(f"[INFO] YOLO predict will use device={PREDICT_KW['device']}")

# Your custom class list (model-dependent)
className = ["License"]

# =========================
# MQTT client setup
# =========================
mqtt_client = mqtt.Client()
if MQTT_USER is not None:
    mqtt_client.username_pw_set(MQTT_USER, MQTT_PASS if MQTT_PASS is not None else "")

def on_connect(client, userdata, flags, rc, properties=None):
    print(f"[MQTT] {'connected' if rc == 0 else f'connect failed rc={rc}'}")

def on_disconnect(client, userdata, rc, properties=None):
    print(f"[MQTT] disconnected rc={rc}")

mqtt_client.on_connect = on_connect
mqtt_client.on_disconnect = on_disconnect

try:
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=30)
except Exception as e:
    print(f"[MQTT] Initial connect failed: {e}")
mqtt_client.loop_start()

_last_servo_publish_ts = 0.0

def publish_servo(angle=90):
    global _last_servo_publish_ts
    now_ts = datetime.now().timestamp()
    if now_ts - _last_servo_publish_ts < SERVO_TRIGGER_COOLDOWN_SEC:
        return
    payload = str(int(angle))  # e.g., "90"
    try:
        mqtt_client.publish(MQTT_TOPIC, payload=payload, qos=MQTT_QOS, retain=False)
        _last_servo_publish_ts = now_ts
        print(f"[MQTT] Published {payload} to {MQTT_TOPIC}")
    except Exception as e:
        print(f"[MQTT] Publish failed: {e}")

# ---------- Stream loop (RTSP via Ultralytics) ----------
startTime = datetime.now()
license_plates = set()
frame_count = 0
window_name = "License Plate Detection (ESC to quit)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

try:
    # Iterate the RTSP stream from YOLO directly; ensures frame/detections match
    for res in model.predict(source=rtsp_url, stream=True, **PREDICT_KW):
        frame = res.orig_img
        if frame is None:
            # transient decode issue; skip
            continue

        frame_count += 1
        if frame_count % 30 == 1:
            print(f"[INFO] Frame Number: {frame_count}")

        # Process detections
        boxes = getattr(res, "boxes", None)
        if boxes is not None:
            for box in boxes:
                # xyxy
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # class + conf
                classNameInt = int(box.cls[0]) if hasattr(box, "cls") else 0
                clsName = className[classNameInt] if classNameInt < len(className) else "License"
                conf = math.ceil(float(box.conf[0]) * 100) / 100 if hasattr(box, "conf") else 0.0

                # OCR on ROI
                label = easyocr_ocr(frame, x1, y1, x2, y2)

                # If a number plate text is detected, trigger servo and push to Firebase once
                currentTime = datetime.now()
                if label:
                    # Send MQTT command to rotate servo 90 degrees
                    publish_servo(angle=90)

                    license_plates.add(label)
                    push_plate_to_firebase_once(label, currentTime.isoformat())

                # Draw label above box
                text = label if label else ""
                textSize = cv2.getTextSize(text, 0, fontScale=0.5, thickness=2)[0]
                c2 = (x1 + textSize[0], max(0, y1 - textSize[1] - 3))
                cv2.rectangle(frame, (x1, max(0, y1 - textSize[1] - 10)), c2, (255, 0, 0), -1)
                cv2.putText(frame, text, (x1, max(10, y1 - 2)), 0, 0.5, (255, 255, 255),
                            thickness=1, lineType=cv2.LINE_AA)

        # Periodically clear in-memory set to allow re-reporting after a window
        if (datetime.now() - startTime).seconds >= 20:
            startTime = datetime.now()
            license_plates.clear()

        # Display
        cv2.imshow(window_name, frame)
        # ESC to quit
        if (cv2.waitKey(1) & 0xFF) == 27:
            break

except KeyboardInterrupt:
    pass
finally:
    cv2.destroyAllWindows()
    try:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
    except Exception:
        pass
