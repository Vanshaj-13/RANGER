import time
import datetime
import cv2
from ultralytics import YOLO

import firebase_admin
from firebase_admin import credentials, db


# ----- Firebase setup -----
cred = credentials.Certificate("ranger_new.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://ranger-e0a95-default-rtdb.firebaseio.com/"
})
detections_ref = db.reference("/detections")


def now_iso():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


# ----- Load YOLO model -----
# Use an Ultralytics model file; update as needed ("yolov8n.pt", "yolo11n.pt", etc.)
# If you truly need YOLOv5s from Ultralytics' wrapper, ensure it's compatible with your install
model = YOLO("yolov8n.pt")

# Class names
names = model.model.names if hasattr(model.model, "names") else model.names

# ----- Target labels and thresholds -----
animal_labels = {"dog", "cat", "bird", "horse", "sheep", "cow"}
target_labels = {"person", "car"} | animal_labels
conf_threshold = 0.25  # optional: you can pass conf=... to predict() as well


# ----- IoU + simple tracker -----
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
        self.active_tracks = {}  # tid -> {'bbox': [x1,y1,x2,y2], 'last': ts}
        self.seen_tracks = set()

    def match_or_create(self, bbox):
        """
        Returns (track_id, is_new_track)
        """
        now = time.time()

        # Expire old tracks
        for tid in list(self.active_tracks.keys()):
            if now - self.active_tracks[tid]['last'] > self.ttl:
                del self.active_tracks[tid]

        # Find best IoU match
        best_tid, best_iou = None, 0.0
        for tid, t in self.active_tracks.items():
            i = iou(bbox, t['bbox'])
            if i > best_iou:
                best_tid, best_iou = tid, i

        # Update or create
        if best_iou >= self.iou_thr and best_tid is not None:
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


# ----- Stream source -----
# Replace with your Pi's RTSP URL:
rtsp_url = "http://192.168.0.242:8080/stream.mjpg"

# Optional: reduce load by setting imgsz, conf, device, half, etc.
# stream=True returns a generator of Results objects for each frame
frame_id = 0
window_name = "YOLO Detections (ESC to quit)"

# Create a named window (helps with key events)
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

try:
    # Iterate over results from the stream
    # Note: Ultralytics will handle decoding and frame retrieval internally
    for res in model.predict(
        source=rtsp_url,
        stream=True,
        conf=conf_threshold,   # filter low-confidence on the model side
        verbose=False
    ):
        frame_id += 1

        # The original BGR frame for drawing
        frame = res.orig_img
        if frame is None:
            # In case of intermittent decoding issues, skip this iteration
            continue
        annotated = frame.copy()

        if not hasattr(res, "boxes") or res.boxes is None:
            # No detections this frame
            cv2.imshow(window_name, annotated)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        # Iterate detected boxes
        for box in res.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            # Model-level conf already applied, but we can enforce again if desired
            if conf < conf_threshold:
                continue

            label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]

            # Only track and push target labels
            if label not in target_labels:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            obj_type = "person" if label == "person" else ("car" if label == "car" else "animal")

            # Select tracker
            if obj_type == "person":
                tid, is_new = person_tracker.match_or_create([x1, y1, x2, y2])
            elif obj_type == "car":
                tid, is_new = car_tracker.match_or_create([x1, y1, x2, y2])
            else:
                tid, is_new = animal_tracker.match_or_create([x1, y1, x2, y2])

            # Push only on first appearance of a track
            if is_new:
                det = {
                    "type": obj_type,
                    "raw_label": label,
                    "time": now_iso(),
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                    "frame_id": frame_id,
                    "track_id": tid
                }
                try:
                    detections_ref.push(det)
                except Exception as e:
                    # Log and continue; do not crash on transient Firebase errors
                    print(f"Firebase push failed: {e}")

            # Draw
            color = (0, 200, 0) if obj_type == "person" else ((0, 140, 255) if obj_type == "car" else (255, 0, 0))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            caption = f"{label} {conf:.2f} id:{tid}"
            (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, caption, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show window
        cv2.imshow(window_name, annotated)
        # Use 1ms wait for streaming; ESC to quit
        if cv2.waitKey(1) & 0xFF == 27:
            break

except KeyboardInterrupt:
    pass
finally:
    cv2.destroyAllWindows()
