#!/usr/bin/env python3
"""
Ranger MQTT Controller with in-app MJPEG preview (no VLC/UDP)
Requirements: PyQt6, paho-mqtt, requests
"""

import sys, re
import threading
from PyQt6 import QtCore, QtGui, QtWidgets
import paho.mqtt.client as mqtt
import requests

# Defaults
BROKER_DEFAULT = "127.0.0.1"
PORT_DEFAULT   = 1883
TOPIC_DEFAULT  = "ranger/control/dir"
STREAM_URL_DEFAULT = "http://192.168.0.242:8080/stream.mjpg"  # change to your Pi

# -------- MJPEG support --------
class MjpegStreamThread(QtCore.QThread):
    frameReady = QtCore.pyqtSignal(QtGui.QImage)
    error = QtCore.pyqtSignal(str)

    def __init__(self, url: str, timeout: float = 6.0, parent=None):
        super().__init__(parent)
        self.url = url
        self.timeout = timeout
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        try:
            with requests.get(
                self.url,
                stream=True,
                timeout=self.timeout,
                headers={"Accept": "multipart/x-mixed-replace"}
            ) as r:
                r.raise_for_status()
                ctype = r.headers.get("Content-Type", "")
                m = re.search(r'boundary=([^;]+)', ctype, re.IGNORECASE)
                boundary = m.group(1).strip() if m else "frame"
                if not boundary.startswith("--"):
                    boundary = "--" + boundary
                boundary_b = boundary.encode()

                buf = b""
                for chunk in r.iter_content(chunk_size=4096):
                    if self._stop.is_set():
                        break
                    if not chunk:
                        continue
                    buf += chunk
                    while True:
                        soi = buf.find(b"\xff\xd8")
                        eoi = buf.find(b"\xff\xd9", soi + 2)
                        if soi != -1 and eoi != -1:
                            jpg = buf[soi:eoi+2]
                            buf = buf[eoi+2:]
                            img = QtGui.QImage.fromData(jpg, "JPG")
                            if not img.isNull():
                                self.frameReady.emit(img)
                        else:
                            if len(buf) > 2_000_000:
                                bpos = buf.find(boundary_b)
                                if bpos > 0:
                                    buf = buf[bpos:]
                            break
        except Exception as e:
            self.error.emit(str(e))

class MjpegViewer(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread = None
        self._url = ""

        self.view = QtWidgets.QLabel("No video")
        self.view.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.view.setMinimumSize(320, 240)
        self.view.setStyleSheet("background:#111; color:#bbb; border:1px solid #333;")

        self.startBtn = QtWidgets.QPushButton("Start Preview")
        self.stopBtn  = QtWidgets.QPushButton("Stop")
        self.stopBtn.setEnabled(False)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.startBtn)
        row.addWidget(self.stopBtn)
        row.addStretch(1)

        v = QtWidgets.QVBoxLayout(self)
        v.addWidget(self.view)
        v.addLayout(row)

        self.startBtn.clicked.connect(self._start)
        self.stopBtn.clicked.connect(self._stop)

    def setUrl(self, url: str):
        self._url = url

    def _start(self):
        if not self._url.lower().startswith("http"):
            QtWidgets.QMessageBox.warning(self, "Unsupported URL",
                "In-app preview supports MJPEG over HTTP.\nExample: http://<pi-ip>:8080/stream.mjpg")
            return
        self._stop()
        self._thread = MjpegStreamThread(self._url)
        self._thread.frameReady.connect(self._on_frame)
        self._thread.error.connect(self._on_error)
        self._thread.start()
        self.startBtn.setEnabled(False)
        self.stopBtn.setEnabled(True)

    def _stop(self):
        if self._thread and self._thread.isRunning():
            self._thread.stop()
            self._thread.wait(1000)
        self._thread = None
        self.startBtn.setEnabled(True)
        self.stopBtn.setEnabled(False)

    def closeEvent(self, e: QtGui.QCloseEvent):
        self._stop()
        super().closeEvent(e)

    @QtCore.pyqtSlot(QtGui.QImage)
    def _on_frame(self, img: QtGui.QImage):
        pix = QtGui.QPixmap.fromImage(img)
        self.view.setPixmap(pix.scaled(
            self.view.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation
        ))

    @QtCore.pyqtSlot(str)
    def _on_error(self, msg: str):
        QtWidgets.QMessageBox.warning(self, "Stream error", msg)
        self._stop()

# -------- Your controller --------
class MqttController(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ranger MQTT Controller")
        self.setMinimumWidth(700)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

        self.hostEdit  = QtWidgets.QLineEdit(BROKER_DEFAULT)
        self.portSpin  = QtWidgets.QSpinBox()
        self.portSpin.setRange(1, 65535); self.portSpin.setValue(PORT_DEFAULT)
        self.topicEdit = QtWidgets.QLineEdit(TOPIC_DEFAULT)
        self.connectBtn = QtWidgets.QPushButton("Connect")
        self.statusLbl  = QtWidgets.QLabel("Disconnected")
        self.statusLbl.setStyleSheet("color:#ff6b6b; font-weight:600")

        self.streamUrlEdit = QtWidgets.QLineEdit(STREAM_URL_DEFAULT)
        self.previewBtn    = QtWidgets.QPushButton("Preview in App")

        form = QtWidgets.QFormLayout()
        form.addRow("Broker host:", self.hostEdit)
        form.addRow("Broker port:", self.portSpin)
        form.addRow("Command topic:", self.topicEdit)
        form.addRow("Stream URL:", self.streamUrlEdit)

        streamControls = QtWidgets.QHBoxLayout()
        streamControls.addWidget(self.previewBtn)
        streamControls.addStretch(1)

        grid = QtWidgets.QGridLayout()
        btnUp    = QtWidgets.QPushButton("↑ Forward")
        btnDown  = QtWidgets.QPushButton("↓ Reverse")
        btnLeft  = QtWidgets.QPushButton("← Left")
        btnRight = QtWidgets.QPushButton("→ Right")
        btnBrake = QtWidgets.QPushButton("⎵ Brake (Space)")
        btnCoast = QtWidgets.QPushButton("C Coast")
        for b in (btnUp, btnDown, btnLeft, btnRight, btnBrake, btnCoast):
            b.setMinimumSize(120, 48)
        grid.addWidget(btnUp,    0, 1)
        grid.addWidget(btnLeft,  1, 0)
        grid.addWidget(btnBrake, 1, 1)
        grid.addWidget(btnRight, 1, 2)
        grid.addWidget(btnDown,  2, 1)
        grid.addWidget(btnCoast, 2, 2)

        self.logBox = QtWidgets.QPlainTextEdit()
        self.logBox.setReadOnly(True)
        self.log("Ready. Arrow keys to drive. Space = brake, C = coast.")
        self.log("Use 'Preview in App' for MJPEG (http://.../stream.mjpg).")

        self.viewer = MjpegViewer()

        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.connectBtn)
        row.addStretch(1)
        row.addWidget(self.statusLbl)

        v = QtWidgets.QVBoxLayout(self)
        v.addLayout(form)
        v.addLayout(row)
        v.addLayout(streamControls)
        v.addWidget(self.viewer)
        v.addSpacing(8)
        v.addLayout(grid)
        v.addSpacing(8)
        v.addWidget(self.logBox)

        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, protocol=mqtt.MQTTv5)
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect

        self.connectBtn.clicked.connect(self.toggle_connect)
        btnUp.clicked.connect(lambda: self.publish_cmd("forward"))
        btnDown.clicked.connect(lambda: self.publish_cmd("reverse"))
        btnLeft.clicked.connect(lambda: self.publish_cmd("left"))
        btnRight.clicked.connect(lambda: self.publish_cmd("right"))
        btnBrake.clicked.connect(lambda: self.publish_cmd("brake"))
        btnCoast.clicked.connect(lambda: self.publish_cmd("coast"))
        self.previewBtn.clicked.connect(self.preview_stream)

        self._pressed = set()
        self.mqtt_timer = QtCore.QTimer(self)
        self.mqtt_timer.timeout.connect(lambda: self.client.loop(timeout=0.05))
        self.mqtt_timer.setInterval(20)  # ~50 FPS

    # ----- MQTT -----
    def toggle_connect(self):
        if getattr(self.client, "_sock", None):
            try: self.client.disconnect()
            except Exception: pass
            return
        host = self.hostEdit.text().strip()
        port = int(self.portSpin.value())
        self.log(f"Connecting to {host}:{port} ...")
        try:
            self.client.connect(host, port=port, keepalive=30)
            self.mqtt_timer.start()
        except Exception as e:
            self.status_bad(f"Connect error: {e}")
            self.log(f"Connect error: {e}")

    def on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            self.status_ok("Connected")
            self.connectBtn.setText("Disconnect")
            self.log("MQTT connected.")
        else:
            self.status_bad(f"Connect failed: {reason_code}")
            self.log(f"Connect failed: {reason_code}")

    def on_disconnect(self, client, userdata, rc, properties):
        self.status_bad("Disconnected")
        self.connectBtn.setText("Connect")
        self.mqtt_timer.stop()
        self.log(f"MQTT disconnected (rc={rc}).")

    def publish_cmd(self, cmd: str):
        topic = self.topicEdit.text().strip()
        if not topic:
            self.log("No topic set.")
            return
        try:
            self.client.publish(topic, cmd, qos=0)
            self.log(f"→ {topic} : {cmd}")
        except Exception as e:
            self.log(f"Publish error: {e}")

    # ----- UI helpers -----
    def status_ok(self, msg):
        self.statusLbl.setText(msg)
        self.statusLbl.setStyleSheet("color:#38d9a9; font-weight:600")

    def status_bad(self, msg):
        self.statusLbl.setText(msg)
        self.statusLbl.setStyleSheet("color:#ff6b6b; font-weight:600")

    def log(self, s):
        self.logBox.appendPlainText(s)

    # ----- Keys -----
    def keyPressEvent(self, e: QtGui.QKeyEvent):
        key = e.key()
        if e.isAutoRepeat(): return
        self._pressed.add(key)
        if   key == QtCore.Qt.Key.Key_Up:    self.publish_cmd("forward")
        elif key == QtCore.Qt.Key.Key_Down:  self.publish_cmd("reverse")
        elif key == QtCore.Qt.Key.Key_Left:  self.publish_cmd("left")
        elif key == QtCore.Qt.Key.Key_Right: self.publish_cmd("right")
        elif key == QtCore.Qt.Key.Key_Space: self.publish_cmd("brake")
        elif key == QtCore.Qt.Key.Key_C:     self.publish_cmd("coast")
        else: super().keyPressEvent(e)

    def keyReleaseEvent(self, e: QtGui.QKeyEvent):
        key = e.key()
        if e.isAutoRepeat(): return
        if key in self._pressed: self._pressed.remove(key)
        if key in (QtCore.Qt.Key.Key_Up, QtCore.Qt.Key.Key_Down,
                   QtCore.Qt.Key.Key_Left, QtCore.Qt.Key.Key_Right):
            self.publish_cmd("coast")
        else:
            super().keyReleaseEvent(e)

    # ----- URL helper (STATIC) -----
    @staticmethod
    def resolve_mjpeg_url(url: str) -> str:
        u = url.strip()
        if not u.lower().startswith("http"):
            return u
        u = u.rstrip("/")
        if u.endswith("/stream.mjpg") or u.endswith("/?action=stream"):
            return u
        # default to Flask/OpenCV style; change to "/?action=stream" for mjpg-streamer by default
        return u + "/stream.mjpg"

    # ----- Stream -----
    def preview_stream(self):
        url = self.streamUrlEdit.text().strip()
        url = self.resolve_mjpeg_url(url)  # <-- fixed call
        self.viewer.setUrl(url)
        self.viewer._start()

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MqttController()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
