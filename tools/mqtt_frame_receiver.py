#!/usr/bin/env python3
"""
ESP32-CAM MQTT event receiver with HTTP frame pull.

Listens for motion events over MQTT, then fetches the JPEG directly
from the ESP32's HTTP server via GET /control?still=1.

This is faster and more reliable than pushing binary data through MQTT.

Usage:
    pip install paho-mqtt requests
    python3 mqtt_frame_receiver.py --broker 192.168.1.x --camera 192.168.1.y

The ESP32-CAM firmware publishes motion events to:
    {topic_prefix}sensor/{hostname}/motion  -> on / off
"""

import argparse
import os
import sys
import time
import traceback
import threading
import requests
import paho.mqtt.client as mqtt

# ---------------------------------------------------------------------------
# Configuration — override via CLI args or edit defaults here
# ---------------------------------------------------------------------------
DEFAULT_BROKER       = "localhost"
DEFAULT_PORT         = 1883
DEFAULT_TOPIC_PREFIX = "homeassistant/"
DEFAULT_OUTPUT_DIR   = "./motion_frames"
HTTP_TIMEOUT         = 5   # seconds for the HTTP still request
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="ESP32-CAM MQTT event receiver + HTTP frame puller")
    p.add_argument("--broker",   default=DEFAULT_BROKER,        help="MQTT broker IP/hostname")
    p.add_argument("--port",     default=DEFAULT_PORT, type=int, help="MQTT broker port (default 1883)")
    p.add_argument("--prefix",   default=DEFAULT_TOPIC_PREFIX,  help="mqtt_topic_prefix set in ESP32 config")
    p.add_argument("--hostname", default="",                     help="ESP32 hostname (omit to accept any device)")
    p.add_argument("--camera",   required=True,                  help="ESP32 camera IP or hostname for HTTP requests")
    p.add_argument("--outdir",   default=DEFAULT_OUTPUT_DIR,     help="Directory to save received frames")
    p.add_argument("--user",     default="",                     help="MQTT broker username (if required)")
    p.add_argument("--password", default="",                     help="MQTT broker password (if required)")
    return p.parse_args()


def fetch_frame(camera_ip, out_dir):
    """Fetch a JPEG from the ESP32 HTTP server and save it."""
    url = f"http://{camera_ip}/control?still=1"
    try:
        t0 = time.time()
        resp = requests.get(url, timeout=HTTP_TIMEOUT)
        elapsed = time.time() - t0
        if resp.status_code == 200 and resp.content:
            ts = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(out_dir, f"motion_{ts}.jpg")
            with open(filename, "wb") as f:
                f.write(resp.content)
            print(f"[+] Frame saved: {filename}  ({len(resp.content)} bytes, {elapsed:.2f}s)")
            # Hook: call your YOLO pipeline here, e.g.:
            #   run_yolo(filename)
        else:
            print(f"[!] HTTP {resp.status_code} from {url}")
    except requests.exceptions.Timeout:
        print(f"[!] Timeout fetching frame from {url}")
    except Exception as e:
        print(f"[!] Error fetching frame: {e}")


def hostname_from_topic(topic):
    parts = topic.split("/")
    try:
        return parts[parts.index("sensor") + 1]
    except (ValueError, IndexError):
        return "unknown"


def on_connect(client, userdata, flags, rc, *args):
    print(f"[dbg] on_connect rc={rc!r}")
    if rc == 0:
        motion_topic = userdata["motion_topic"]
        result = client.subscribe(motion_topic)
        print(f"[+] Connected. Subscribed to {motion_topic}  result={result}")
    else:
        print(f"[!] Connection failed, rc={rc}")


def on_message(client, userdata, msg):
    try:
        hostname = hostname_from_topic(msg.topic)
        payload  = msg.payload.decode("utf-8", errors="replace").strip()
        print(f"[motion/{hostname}] {payload}")

        if payload.lower() == "on":
            # Fetch the frame in a background thread so we don't block the MQTT loop
            t = threading.Thread(
                target=fetch_frame,
                args=(userdata["camera_ip"], userdata["out_dir"]),
                daemon=True,
            )
            t.start()

    except Exception:
        print("[!] Exception in on_message:", file=sys.stderr)
        traceback.print_exc()


def on_callback_exception(client, userdata, callback, exception):
    print(f"[!] Exception in paho callback {callback.__name__}: {exception}", file=sys.stderr)
    traceback.print_exception(type(exception), exception, exception.__traceback__)


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    device = args.hostname if args.hostname else "+"
    motion_topic = f"{args.prefix}sensor/{device}/motion"

    userdata = {
        "motion_topic": motion_topic,
        "camera_ip":    args.camera,
        "out_dir":      args.outdir,
    }

    if hasattr(mqtt, "CallbackAPIVersion"):
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, userdata=userdata)
    else:
        client = mqtt.Client(userdata=userdata)

    client.on_connect = on_connect
    client.on_message = on_message
    if hasattr(client, "on_callback_exception"):
        client.on_callback_exception = on_callback_exception

    if args.user:
        client.username_pw_set(args.user, args.password)

    print(f"[*] Connecting to MQTT broker at {args.broker}:{args.port} ...")
    print(f"[*] Will fetch frames from http://{args.camera}/control?still=1")
    client.connect(args.broker, args.port, keepalive=60)

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n[*] Stopped.")
        client.disconnect()


if __name__ == "__main__":
    main()
