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


FIREBASE_CREDENTIALS_PATH = r"ranger_database.json"
FIREBASE_DB_URL = "https://ranger-e0a95-default-rtdb.firebaseio.com/"


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


try:
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        
        pass
except Exception:
    pass


USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if USE_CUDA else "cpu"
print(f"[INFO] Torch CUDA available: {USE_CUDA} -> using {DEVICE}")

def init_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
        firebase_admin.initialize_app(cred, {
            "databaseURL": FIREBASE_DB_URL
        })
    
    return db.reference("/license_events"), db.reference("/license_events_unique")


cap = cv2.VideoCapture("carLicence4.mp4")


model = YOLO("best.pt")


try:
    if USE_CUDA:
        
        model.to('cuda')
        print("[INFO] YOLO model moved to CUDA.")
    else:
        print("[INFO] YOLO model running on CPU.")
except Exception as e:
    print(f"[WARN] Could not move YOLO model to CUDA explicitly: {e}. It may still use GPU via device='cuda' during predict.")


count = 0


className = ["License"]


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


fb_events_ref, fb_unique_ref = init_firebase()
sent_to_firebase = set()  

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
        pass

load_existing_plates_once()

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
                img,
                detail=1,
                paragraph=False,
                batch_size=1,
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
            if not isinstance(conf, (float, int)):
                continue
            cleaned = re.sub(r"[^A-Z0-9-]", "", text.upper())
            if len(cleaned) >= 5 and conf > best_conf:
                best_text, best_conf = cleaned, float(conf)

    
    if best_conf < 0.45:
        return ""

    
    best_text = best_text.replace("O", "0")
    return best_text


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
        
        sent_to_firebase.add(plate_text)
        return

startTime = datetime.now()
license_plates = set()


PREDICT_KW = {"conf": 0.45, "verbose": False}

PREDICT_KW["device"] = 0 if USE_CUDA else "cpu" 

print(f"[INFO] YOLO predict will use device={PREDICT_KW['device']}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    currentTime = datetime.now()
    count += 1
    if count % 30 == 1:
        print(f"[INFO] Frame Number: {count}")

    
    results = model.predict(frame, **PREDICT_KW)

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            
            classNameInt = int(box.cls[0]) if hasattr(box, "cls") else 0
            clsName = className[classNameInt] if classNameInt < len(className) else "License"
            conf = math.ceil(float(box.conf[0]) * 100) / 100 if hasattr(box, "conf") else 0.0

            
            label = easyocr_ocr(frame, x1, y1, x2, y2)

            if label:
                
                license_plates.add(label)
                
                push_plate_to_firebase_once(label, currentTime.isoformat())

            
            text = label if label else ""
            textSize = cv2.getTextSize(text, 0, fontScale=0.5, thickness=2)[0]
            c2 = (x1 + textSize[0], max(0, y1 - textSize[1] - 3))
            cv2.rectangle(frame, (x1, max(0, y1 - textSize[1] - 10)), c2, (255, 0, 0), -1)
            cv2.putText(frame, text, (x1, max(10, y1 - 2)), 0, 0.5, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    
    if (currentTime - startTime).seconds >= 20:
        startTime = currentTime
        license_plates.clear()

    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('1'):
        break

cap.release()
cv2.destroyAllWindows()
