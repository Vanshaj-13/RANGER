
import os
import time
import keyboard  # pip install keyboard
from paho.mqtt import client as mqtt  # pip install paho-mqtt

# ===== MQTT config =====
BROKER_HOST = os.getenv("MQTT_HOST", "192.168.1.10")  # broker IP
BROKER_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_USER = os.getenv("MQTT_USER", None)
MQTT_PASS = os.getenv("MQTT_PASS", None)
CLIENT_ID = os.getenv("MQTT_CLIENT_ID", "pc-arm-controller")

# ===== Topics =====
TOPIC_SERVO1_STEP = "ranger/control/arm/servo1/step"
TOPIC_SERVO2_STEP = "ranger/control/arm/servo2/step"
TOPIC_CENTER      = "ranger/control/arm/center"

# Step size in degrees per press
STEP_DEG = int(os.getenv("STEP_DEG", "5"))

def connect_mqtt():
    c = mqtt.Client(client_id=CLIENT_ID, protocol=mqtt.MQTTv311)
    if MQTT_USER and MQTT_PASS:
        c.username_pw_set(MQTT_USER, MQTT_PASS)
    c.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
    return c

def main():
    client = connect_mqtt()
    print(f"Connected to MQTT broker at {BROKER_HOST}")
    print("Controls: Right=Servo1+, Left=Servo2-, Up=Servo2+, Down=Servo1-, C=center, Q=quit")

    try:
        while True:
            if keyboard.is_pressed("right"):
                client.publish(TOPIC_SERVO1_STEP, str(+STEP_DEG))
                print(f"→ Servo1 +{STEP_DEG}")
                time.sleep(0.12)

            elif keyboard.is_pressed("left"):
                client.publish(TOPIC_SERVO2_STEP, str(-STEP_DEG))
                print(f"← Servo2 -{STEP_DEG}")
                time.sleep(0.12)

            elif keyboard.is_pressed("up"):
                client.publish(TOPIC_SERVO2_STEP, str(+STEP_DEG))
                print(f"↑ Servo2 +{STEP_DEG}")
                time.sleep(0.12)

            elif keyboard.is_pressed("down"):
                client.publish(TOPIC_SERVO1_STEP, str(-STEP_DEG))
                print(f"↓ Servo1 -{STEP_DEG}")
                time.sleep(0.12)

            elif keyboard.is_pressed("c"):
                client.publish(TOPIC_CENTER, "1")
                print("[Center] Both servos to 90°")
                time.sleep(0.2)

            elif keyboard.is_pressed("q"):
                break

            time.sleep(0.01)

    finally:
        client.disconnect()
        print("Stopped publisher.")

if __name__ == "__main__":
    main()
