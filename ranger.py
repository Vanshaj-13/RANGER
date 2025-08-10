import time
import datetime
import cv2
from ultralytics import YOLO

import firebase_admin
from firebase_admin import credentials, db

cred = credentials.Certificate("ranger_database.json")  
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://ranger-e0a95-default-rtdb.firebaseio.com/"  
})
detections_ref = db.reference("/detections")

def now_iso():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


model = YOLO("yolov5s.pt")

names = model.model.names if hasattr(model.model, 'names') else model.names

animal_labels = {"dog", "cat", "bird", "horse", "sheep", "cow"}
target_labels = {"person", "car"} | animal_labels

conf_threshold = 0.25

video_path = "test_video.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
delay_ms = int(1000 / fps) if fps and fps > 0 else 1

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0

class SimpleIoUTracker:
    def __init__(self, iou_thr=0.5, ttl=3.0):
        self.iou_thr = iou_thr
        self.ttl = ttl
        self.next_track_id = 1
        self.active_tracks = {}  
        self.seen_tracks = set()

    def match_or_create(self, bbox):
        """
        Returns (track_id, is_new_track)
        """
        now = time.time()


        for tid in list(self.active_tracks.keys()):
            if now - self.active_tracks[tid]['last'] > self.ttl:
                del self.active_tracks[tid]


        best_tid, best_iou = None, 0.0
        for tid, t in self.active_tracks.items():
            i = iou(bbox, t['bbox'])
            if i > best_iou:
                best_tid, best_iou = tid, i

        if best_iou >= self.iou_thr:
            
            self.active_tracks[best_tid]['bbox'] = bbox
            self.active_tracks[best_tid]['last'] = now
            return best_tid, False
        else:
            
            tid = self.next_track_id
            self.next_track_id += 1
            self.active_tracks[tid] = {'bbox': bbox, 'last': now}
            self.seen_tracks.add(tid)
            return tid, True


TRACK_CFG = {
    "person": {"iou_thr": 0.5, "ttl": 3.0},
    "car":    {"iou_thr": 0.6, "ttl": 3.0},  
    "animal": {"iou_thr": 0.5, "ttl": 3.0},
}

person_tracker = SimpleIoUTracker(**TRACK_CFG["person"])
car_tracker    = SimpleIoUTracker(**TRACK_CFG["car"])
animal_tracker = SimpleIoUTracker(**TRACK_CFG["animal"])


frame_id = 0
window_name = "YOLO Detections (ESC to quit)"

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or failed to read frame.")
        break
    frame_id += 1

    results = model.predict(frame, conf=conf_threshold, verbose=False)
    res = results[0]
    annotated = frame.copy()

    for box in res.boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]

        if label not in target_labels:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        obj_type = "person" if label == "person" else ("car" if label == "car" else "animal")

        track_tag = ""
        should_push = False
        tid = None

        if obj_type == "person":
            tid, is_new = person_tracker.match_or_create([x1, y1, x2, y2])
            track_tag = f" id:{tid}"
            should_push = is_new  

        elif obj_type == "car":
            tid, is_new = car_tracker.match_or_create([x1, y1, x2, y2])
            track_tag = f" id:{tid}"
            should_push = is_new  

        else:  # animal
            tid, is_new = animal_tracker.match_or_create([x1, y1, x2, y2])
            track_tag = f" id:{tid}"
            should_push = is_new  
        if should_push:
            det = {
                "type": obj_type,
                "raw_label": label,
                "time": now_iso(),
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "frame_id": frame_id,
                "track_id": tid
            }
            detections_ref.push(det)

        
        color = (0, 200, 0) if obj_type == "person" else ((0, 140, 255) if obj_type == "car" else (255, 0, 0))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        caption = f"{label} {conf:.2f}{track_tag}"
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, caption, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow(window_name, annotated)
    key = cv2.waitKey(delay_ms) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()


