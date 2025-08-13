from time import sleep
from gpiozero import DigitalOutputDevice
import paho.mqtt.client as mqtt
import os
import signal
import sys

# Motor A pins (BCM numbering)
IN1 = DigitalOutputDevice(17, active_high=True, initial_value=False)  # IN1
IN2 = DigitalOutputDevice(25, active_high=True, initial_value=False)  # IN2
IN3 = DigitalOutputDevice(27, active_high=True, initial_value=False)  #IN3
IN4 = DigitalOutputDevice(24, active_high=True, initial_value=False)  #IN4

def left():
    IN1.on();  IN2.off()
    IN3.on();  IN4.off()

def right():
    IN1.off(); IN2.on()
    IN3.off(); IN4.on()

def forward():     # pivot right in place
    IN1.off(); IN2.on()   # left side reverse
    IN3.on();  IN4.off()  # right side forward

def reverse():    # pivot left in place
    IN1.on();  IN2.off()  # left side forward
    IN3.off(); IN4.on()   # right side reverse

def brake():
    IN1.on(); IN2.on()
    IN3.on(); IN4.on()

def coast():
    IN1.off(); IN2.off()
    IN3.off(); IN4.off()

def handle_command(cmd: str):
    c = cmd.strip().lower()
    if c == "forward":
        forward()
    elif c == "reverse" or c == "back" or c == "backward" or c == "backwards":
        reverse()
    elif c == "left":
        left()
    elif c == "right":
        right()
    elif c == "brake" or c == "stop":
        brake()
    elif c == "coast":
        coast()
    else:
        print(f"Unknown command: {cmd!r}")

# ---- MQTT setup ----
BROKER = "192.168.2.242"   # <-- REPLACE with your Mac's IP from `ipconfig getifaddr en0`
TOPIC  = "ranger/control/dir"
CLIENT_ID = "ranger-1" + os.uname().nodename

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("MQTT connected.")
        client.subscribe(TOPIC, qos=1)
        # Let controller know we're online
        client.publish("robots/rover1/status", "online", qos=1, retain=True)
    else:
        print("MQTT connect failed:", rc)

def on_message(client, userdata, msg):
    payload = msg.payload.decode("utf-8", errors="ignore")
    print(f"[MQTT] {msg.topic}: {payload}")
    handle_command(payload)

def on_disconnect(client, userdata, rc, properties=None):
    print("MQTT disconnected:", rc)

def cleanup_and_exit(*_):
    try:
        coast()
    finally:
        sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, cleanup_and_exit)
    signal.signal(signal.SIGTERM, cleanup_and_exit)

    client = mqtt.Client(client_id=CLIENT_ID, protocol=mqtt.MQTTv5)
    # Optional: set LWT so controllers know if we drop
    client.will_set("robots/rover1/status", "offline", qos=1, retain=True)

    # If you enabled auth on the broker, set credentials:
    # client.username_pw_set("YOUR_USER", "YOUR_PASS")

    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect

    print(f"Connecting to {BROKER} ...")
    client.connect(BROKER, port=1883, keepalive=30)
    client.loop_forever()
