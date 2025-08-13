import os
import json
import time
import paho.mqtt.client as mqtt
from gpiozero import AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory  # needs pigpio daemon running

# ===== MQTT Configuration =====
BROKER_HOST = os.getenv("MQTT_HOST", "192.168.1.10")  # Your broker IP
BROKER_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_USER = os.getenv("MQTT_USER", None)
MQTT_PASS = os.getenv("MQTT_PASS", None)
CLIENT_ID = os.getenv("MQTT_CLIENT_ID", "pi-servo-listener")

# ===== Topics =====
TOPIC_SERVO1_STEP = "ranger/control/arm/servo1/step"
TOPIC_SERVO2_STEP = "ranger/control/arm/servo2/step"
TOPIC_CENTER      = "ranger/control/arm/center"

# ===== Servo Setup =====
SERVO1_PIN = 22  # BCM pin
SERVO2_PIN = 23

MIN_ANGLE = 0
MAX_ANGLE = 180
START_ANGLE = 90

# Pulse widths for most hobby servos
MIN_PW = 0.5 / 1000   # 0.5ms
MAX_PW = 2.5 / 1000   # 2.5ms

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

# Use local pigpio daemon
factory = PiGPIOFactory()
servo1 = AngularServo(SERVO1_PIN, min_angle=MIN_ANGLE, max_angle=MAX_ANGLE,
                      min_pulse_width=MIN_PW, max_pulse_width=MAX_PW,
                      frame_width=0.02, pin_factory=factory)
servo2 = AngularServo(SERVO2_PIN, min_angle=MIN_ANGLE, max_angle=MAX_ANGLE,
                      min_pulse_width=MIN_PW, max_pulse_width=MAX_PW,
                      frame_width=0.02, pin_factory=factory)

angle1 = START_ANGLE
angle2 = START_ANGLE
servo1.angle = angle1
servo2.angle = angle2
print(f"[INIT] Servo1={angle1}°, Servo2={angle2}°")

# ===== MQTT Callbacks =====
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("[MQTT] Connected")
        client.subscribe([
            (TOPIC_SERVO1_STEP, 0),
            (TOPIC_SERVO2_STEP, 0),
            (TOPIC_CENTER, 0)
        ])
        print("[MQTT] Subscribed to topics")
    else:
        print(f"[MQTT] Connection failed: {rc}")

def parse_int(payload: bytes):
    try:
        s = payload.decode().strip()
        if s.startswith("{"):
            obj = json.loads(s)
            return int(obj.get("value"))
        return int(s)
    except Exception:
        return None

def on_message(client, userdata, msg):
    global angle1, angle2
    topic = msg.topic
    val = parse_int(msg.payload)
    print(f"[MSG] {topic} -> {val}")

    if topic == TOPIC_SERVO1_STEP and val is not None:
        angle1 = clamp(angle1 + val, MIN_ANGLE, MAX_ANGLE)
        servo1.angle = angle1
        print(f"Servo1 -> {angle1}°")

    elif topic == TOPIC_SERVO2_STEP and val is not None:
        angle2 = clamp(angle2 + val, MIN_ANGLE, MAX_ANGLE)
        servo2.angle = angle2
        print(f"Servo2 -> {angle2}°")

    elif topic == TOPIC_CENTER:
        angle1 = angle2 = 90
        servo1.angle = servo2.angle = 90
        print("Both servos centered at 90°")

# ===== Main =====
def main():
    client = mqtt.Client(client_id=CLIENT_ID, protocol=mqtt.MQTTv311)
    if MQTT_USER and MQTT_PASS:
        client.username_pw_set(MQTT_USER, MQTT_PASS)
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        servo1.detach()
        servo2.detach()

if __name__ == "__main__":
    main()
