#!/usr/bin/env python3
"""
ESP32-CAM MQTT frame receiver.

Subscribes to the motion-triggered JPEG topic published by the ESP32-CAM
firmware and saves each received frame to disk, ready for YOLO processing.

Usage:
    pip install paho-mqtt
    python3 mqtt_frame_receiver.py --broker 192.168.1.x --hostname esp32cam

The ESP32-CAM firmware publishes a JPEG to:
    {topic_prefix}sensor/{hostname}/still
on every motion detection start.

It also publishes JSON motion events to:
    {topic_prefix}sensor/{hostname}/state   -> {"MOTION":"ON"/"OFF", "TIME":"..."}
"""

import argparse
import os
import time
import paho.mqtt.client as mqtt

# ---------------------------------------------------------------------------
# Configuration — override via CLI args or edit defaults here
# ---------------------------------------------------------------------------
DEFAULT_BROKER      = "localhost"
DEFAULT_PORT        = 1883
DEFAULT_TOPIC_PREFIX = "homeassistant/"
DEFAULT_HOSTNAME    = "esp32cam"
DEFAULT_OUTPUT_DIR  = "./motion_frames"
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="ESP32-CAM MQTT frame receiver")
    p.add_argument("--broker",  default=DEFAULT_BROKER,       help="MQTT broker IP/hostname")
    p.add_argument("--port",    default=DEFAULT_PORT, type=int, help="MQTT broker port (default 1883)")
    p.add_argument("--prefix",  default=DEFAULT_TOPIC_PREFIX, help="mqtt_topic_prefix set in ESP32 config")
    p.add_argument("--hostname",default=DEFAULT_HOSTNAME,      help="ESP32 hostname set in its config")
    p.add_argument("--outdir",  default=DEFAULT_OUTPUT_DIR,    help="Directory to save received frames")
    p.add_argument("--user",    default="",                    help="MQTT broker username (if required)")
    p.add_argument("--password",default="",                    help="MQTT broker password (if required)")
    return p.parse_args()


def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        image_topic  = userdata["image_topic"]
        motion_topic = userdata["motion_topic"]
        client.subscribe(image_topic)
        client.subscribe(motion_topic)
        print(f"[+] Connected to broker. Subscribed to:")
        print(f"    {image_topic}")
        print(f"    {motion_topic}")
    else:
        print(f"[!] Connection failed, rc={rc}")


def on_message(client, userdata, msg):
    out_dir = userdata["out_dir"]

    if msg.topic == userdata["image_topic"]:
        # Binary JPEG payload
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(out_dir, f"motion_{ts}.jpg")
        with open(filename, "wb") as f:
            f.write(msg.payload)
        print(f"[+] Frame saved: {filename}  ({len(msg.payload)} bytes)")

        # Hook: call your YOLO pipeline here, e.g.:
        #   run_yolo(filename)

    elif msg.topic == userdata["motion_topic"]:
        # JSON status message
        print(f"[motion] {msg.payload.decode('utf-8', errors='replace')}")


def main():
    args = parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    image_topic  = f"{args.prefix}sensor/{args.hostname}/still"
    motion_topic = f"{args.prefix}sensor/{args.hostname}/state"

    userdata = {
        "image_topic":  image_topic,
        "motion_topic": motion_topic,
        "out_dir":      args.outdir,
    }

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, userdata=userdata)
    client.on_connect = on_connect
    client.on_message = on_message

    if args.user:
        client.username_pw_set(args.user, args.password)

    print(f"[*] Connecting to MQTT broker at {args.broker}:{args.port} ...")
    client.connect(args.broker, args.port, keepalive=60)

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n[*] Stopped.")
        client.disconnect()


if __name__ == "__main__":
    main()
