#!/usr/bin/env python3
"""
ESP32-CAM MQTT frame receiver.

Subscribes to the motion-triggered JPEG topic published by the ESP32-CAM
firmware and saves each received frame to disk, ready for YOLO processing.

Usage:
    pip install paho-mqtt
    python3 mqtt_frame_receiver.py --broker 192.168.1.x

    # Optionally pin to a specific device (hostname must match ESP32 config):
    python3 mqtt_frame_receiver.py --broker 192.168.1.x --hostname esp32cam

When --hostname is omitted the receiver uses a single-level MQTT wildcard (+)
and accepts frames from any ESP32-CAM on the broker.  The actual hostname is
extracted from the topic and included in the saved filename.

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
DEFAULT_BROKER       = "localhost"
DEFAULT_PORT         = 1883
DEFAULT_TOPIC_PREFIX = "homeassistant/"
DEFAULT_OUTPUT_DIR   = "./motion_frames"
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="ESP32-CAM MQTT frame receiver")
    p.add_argument("--broker",   default=DEFAULT_BROKER,        help="MQTT broker IP/hostname")
    p.add_argument("--port",     default=DEFAULT_PORT, type=int, help="MQTT broker port (default 1883)")
    p.add_argument("--prefix",   default=DEFAULT_TOPIC_PREFIX,  help="mqtt_topic_prefix set in ESP32 config")
    p.add_argument("--hostname", default="",                     help="ESP32 hostname (omit to accept any device)")
    p.add_argument("--outdir",   default=DEFAULT_OUTPUT_DIR,     help="Directory to save received frames")
    p.add_argument("--user",     default="",                     help="MQTT broker username (if required)")
    p.add_argument("--password", default="",                     help="MQTT broker password (if required)")
    return p.parse_args()


def hostname_from_topic(topic):
    # topic format: {prefix}sensor/{hostname}/{suffix}
    # e.g. homeassistant/sensor/esp32cam-abc/still  -> "esp32cam-abc"
    parts = topic.split("/")
    try:
        sensor_idx = parts.index("sensor")
        return parts[sensor_idx + 1]
    except (ValueError, IndexError):
        return "unknown"


def on_connect(client, userdata, flags, rc, *args):
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
    out_dir  = userdata["out_dir"]
    prefix   = userdata["prefix"]
    hostname = hostname_from_topic(msg.topic)

    # Derive the expected suffix for this topic
    still_suffix  = f"{prefix}sensor/{hostname}/still"
    motion_suffix = f"{prefix}sensor/{hostname}/state"

    if msg.topic == still_suffix:
        # Binary JPEG payload
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(out_dir, f"motion_{hostname}_{ts}.jpg")
        with open(filename, "wb") as f:
            f.write(msg.payload)
        print(f"[+] Frame saved: {filename}  ({len(msg.payload)} bytes)")

        # Hook: call your YOLO pipeline here, e.g.:
        #   run_yolo(filename)

    elif msg.topic == motion_suffix:
        # JSON status message
        print(f"[motion/{hostname}] {msg.payload.decode('utf-8', errors='replace')}")

    else:
        print(f"[?] Unknown topic {msg.topic}  ({len(msg.payload)} bytes)")


def main():
    args = parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    device = args.hostname if args.hostname else "+"  # "+" = MQTT single-level wildcard
    image_topic  = f"{args.prefix}sensor/{device}/still"
    motion_topic = f"{args.prefix}sensor/{device}/state"

    userdata = {
        "image_topic":  image_topic,
        "motion_topic": motion_topic,
        "prefix":       args.prefix,
        "out_dir":      args.outdir,
    }

    # CallbackAPIVersion was introduced in paho-mqtt 2.0
    if hasattr(mqtt, "CallbackAPIVersion"):
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, userdata=userdata)
    else:
        client = mqtt.Client(userdata=userdata)
    client.on_connect = on_connect
    client.on_message = on_message

    if args.user:
        client.username_pw_set(args.user, args.password)

    print(f"[*] Connecting to MQTT broker at {args.broker}:{args.port} ...")
    if not args.hostname:
        print(f"[*] No --hostname given, listening for any ESP32-CAM device")
    client.connect(args.broker, args.port, keepalive=60)

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n[*] Stopped.")
        client.disconnect()


if __name__ == "__main__":
    main()
