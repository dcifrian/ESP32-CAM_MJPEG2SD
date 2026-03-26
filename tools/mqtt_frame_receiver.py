#!/usr/bin/env python3
"""
ESP32-CAM MQTT event receiver with HTTP frame pull or video recording.

Listens for motion events over MQTT, then either:
  - (default) fetches a single JPEG via GET /control?still=1
  - (--video)  records a video from the MJPEG stream at /sustain?video=1,
               stopping when motion ends or --video-timeout seconds elapse.

Also supports LDR (light sensor) readings for cat feeder food-level tracking:
  - Subscribes to {prefix}sensor/{hostname}/ldr and logs readings
  - Use --ldr-trigger to request a reading on startup
  - Use --hostname to target a specific device when sending commands

LDR JSON payload: {"ambient": N, "illuminated": N, "differential": N, "percent": N}

Usage:
    pip install paho-mqtt requests
    pip install opencv-python       # only needed for --video mode

    # Image mode (default):
    python3 mqtt_frame_receiver.py --broker 192.168.1.x --camera 192.168.1.y

    # Video mode:
    python3 mqtt_frame_receiver.py --broker 192.168.1.x --camera 192.168.1.y \\
        --video [--video-timeout 300]

    # Trigger an LDR reading on connect (requires --hostname):
    python3 mqtt_frame_receiver.py --broker 192.168.1.x --camera 192.168.1.y \\
        --hostname esp32cam --ldr-trigger
"""

import argparse
import json
import os
import sys
import time
import traceback
import threading
import requests
import paho.mqtt.client as mqtt

# ---------------------------------------------------------------------------
DEFAULT_BROKER        = "localhost"
DEFAULT_PORT          = 1883
DEFAULT_TOPIC_PREFIX  = "homeassistant/"
DEFAULT_OUTPUT_DIR    = "./motion_frames"
DEFAULT_VIDEO_TIMEOUT = 300   # seconds
HTTP_TIMEOUT          = 5     # seconds for still requests
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="ESP32-CAM MQTT event receiver + HTTP frame/video puller")
    p.add_argument("--broker",         default=DEFAULT_BROKER,        help="MQTT broker IP/hostname")
    p.add_argument("--port",           default=DEFAULT_PORT, type=int, help="MQTT broker port (default 1883)")
    p.add_argument("--prefix",         default=DEFAULT_TOPIC_PREFIX,  help="mqtt_topic_prefix set in ESP32 config")
    p.add_argument("--hostname",       default="",                     help="ESP32 hostname (omit to accept any device)")
    p.add_argument("--camera",         required=True,                  help="ESP32 camera IP for HTTP requests")
    p.add_argument("--outdir",         default=DEFAULT_OUTPUT_DIR,     help="Directory to save files")
    p.add_argument("--user",           default="",                     help="MQTT broker username (if required)")
    p.add_argument("--password",       default="",                     help="MQTT broker password (if required)")
    p.add_argument("--video",          action="store_true",            help="Record video instead of capturing a single frame")
    p.add_argument("--video-timeout",  default=DEFAULT_VIDEO_TIMEOUT, type=int, metavar="SECONDS",
                   help=f"Stop recording after this many seconds even if motion continues (default {DEFAULT_VIDEO_TIMEOUT})")
    p.add_argument("--ldr-trigger",    action="store_true",
                   help="Send an LDR measurement command to the ESP32 immediately on connect (requires --hostname)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Image mode
# ---------------------------------------------------------------------------

def fetch_frame(camera_ip, out_dir):
    """Fetch a single JPEG from /control?still=1 and save it."""
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


# ---------------------------------------------------------------------------
# Video mode
# ---------------------------------------------------------------------------

def _parse_mjpeg_frames(resp):
    """Yield raw JPEG bytes from an MJPEG multipart response using Content-Length.

    Reads exactly as many bytes as the ESP says each frame contains, so the
    parser can never get confused by partial sends or coincidental marker bytes.
    """
    buf = b""
    for chunk in resp.iter_content(chunk_size=8192):
        buf += chunk
        while True:
            # Locate the Content-Length header for the next part
            cl_idx = buf.find(b"Content-Length:")
            if cl_idx < 0:
                break
            eol = buf.find(b"\r\n", cl_idx)
            if eol < 0:
                break
            try:
                frame_len = int(buf[cl_idx + 15 : eol].strip())
            except ValueError:
                buf = buf[cl_idx + 15:]   # skip malformed header, keep searching
                continue
            # Find end of part headers (blank line)
            hdr_end = buf.find(b"\r\n\r\n", cl_idx)
            if hdr_end < 0:
                break
            data_start = hdr_end + 4
            data_end   = data_start + frame_len
            if len(buf) < data_end:
                break   # wait for the rest of the frame to arrive
            yield buf[data_start : data_end]
            buf = buf[data_end:]


def record_video(camera_ip, out_dir, stop_event, max_duration, userdata):
    """Stream MJPEG from /sustain?stream=1 and write to a video file.

    The recording deadline is based on the *last* motion "on" event stored in
    userdata["last_motion_time"], so every new motion event extends the window
    by max_duration seconds from that moment.  "off" events are ignored here.
    Reconnects automatically if the stream stalls.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("[!] opencv-python is required for video mode.  Run: pip install opencv-python", file=sys.stderr)
        return

    url = f"http://{camera_ip}/sustain?stream=1"
    ts = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(out_dir, f"video_{ts}.avi")
    print(f"[*] Recording video → {filename}  (idle-stop after {max_duration}s of no motion)")

    writer = None
    frame_count = 0
    t0 = time.time()
    last_report = t0

    def deadline_reached():
        last = userdata.get("last_motion_time") or t0
        return time.time() - last >= max_duration

    try:
        while not stop_event.is_set() and not deadline_reached():
            resp = None
            try:
                resp = requests.get(url, stream=True, timeout=(HTTP_TIMEOUT, 10))
                userdata["current_response"] = resp
                if resp.status_code != 200:
                    print(f"[!] Stream rejected: HTTP {resp.status_code}  ({url})")
                    break
                ct = resp.headers.get("Content-Type", "")
                if "multipart" not in ct:
                    print(f"[!] Unexpected Content-Type: {ct!r}  ({url})")
                    break
                for jpg in _parse_mjpeg_frames(resp):
                    if stop_event.is_set() or deadline_reached():
                        break
                    arr = np.frombuffer(jpg, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                    if writer is None:
                        h, w = frame.shape[:2]
                        print(f"[*] Stream: {w}x{h}", flush=True)
                        writer = cv2.VideoWriter(
                            filename, cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (w, h)
                        )
                    writer.write(frame)
                    frame_count += 1
                    now = time.time()
                    if now - last_report >= 5:
                        last = userdata.get("last_motion_time") or t0
                        remaining = max(0, max_duration - (now - last))
                        fps = frame_count / (now - t0)
                        print(f"[*] {frame_count} frames  {fps:.1f} fps  deadline in {remaining:.0f}s", flush=True)
                        last_report = now
            except requests.exceptions.ReadTimeout:
                if stop_event.is_set() or deadline_reached():
                    break
                print(f"[*] Stream stalled at {frame_count} frames, reconnecting...", flush=True)
                time.sleep(0.5)
                continue
            except Exception as e:
                if stop_event.is_set() or deadline_reached():
                    break
                print(f"[!] Stream error: {e}", flush=True)
                time.sleep(0.5)
                continue
            finally:
                userdata["current_response"] = None
                if resp is not None:
                    try:
                        resp.close()
                    except Exception:
                        pass
            # Stream ended cleanly; small pause before retry
            if not stop_event.is_set() and not deadline_reached():
                time.sleep(0.2)
    except Exception:
        print("[!] Error during video recording:", file=sys.stderr)
        traceback.print_exc()
    finally:
        if writer:
            writer.release()
        elapsed = time.time() - t0
        if frame_count > 0:
            fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"[+] Video saved: {filename}  ({frame_count} frames, {elapsed:.1f}s, {fps:.1f} fps)")
        else:
            print("[!] No frames captured — nothing saved")


# ---------------------------------------------------------------------------
# MQTT callbacks
# ---------------------------------------------------------------------------

def hostname_from_topic(topic):
    parts = topic.split("/")
    try:
        return parts[parts.index("sensor") + 1]
    except (ValueError, IndexError):
        return "unknown"


def on_connect(client, userdata, flags, rc, *args):
    if rc == 0:
        motion_topic = userdata["motion_topic"]
        ldr_topic    = userdata["ldr_topic"]
        client.subscribe(motion_topic)
        client.subscribe(ldr_topic)
        mode = "video" if userdata["video_mode"] else "image"
        print(f"[+] Connected ({mode} mode). Subscribed to {motion_topic}")
        print(f"[+] Subscribed to LDR topic: {ldr_topic}")
        if userdata.get("ldr_trigger") and userdata.get("cmd_topic"):
            client.publish(userdata["cmd_topic"], "ldr")
            print(f"[*] LDR trigger sent → {userdata['cmd_topic']}")
    else:
        print(f"[!] Connection failed, rc={rc}")


def on_message(client, userdata, msg):
    try:
        hostname = hostname_from_topic(msg.topic)
        payload  = msg.payload.decode("utf-8", errors="replace").strip()

        # LDR reading
        if msg.topic == userdata["ldr_topic"] or msg.topic.endswith("/ldr"):
            try:
                data = json.loads(payload)
                ambient      = data.get("ambient", "?")
                illuminated  = data.get("illuminated", "?")
                differential = data.get("differential", "?")
                percent      = data.get("percent", "?")
                print(f"[ldr/{hostname}] ambient={ambient} illuminated={illuminated} "
                      f"differential={differential} ({percent}%)")
            except json.JSONDecodeError:
                print(f"[ldr/{hostname}] {payload}")
            return

        print(f"[motion/{hostname}] {payload}")

        if payload.lower() == "on":
            if userdata["video_mode"]:
                # Always update the deadline on every motion event
                userdata["last_motion_time"] = time.time()
                if userdata["recording_thread"] is None or not userdata["recording_thread"].is_alive():
                    stop_event = threading.Event()
                    userdata["stop_event"] = stop_event
                    t = threading.Thread(
                        target=record_video,
                        args=(
                            userdata["camera_ip"],
                            userdata["out_dir"],
                            stop_event,
                            userdata["video_timeout"],
                            userdata,
                        ),
                        daemon=True,
                    )
                    userdata["recording_thread"] = t
                    t.start()
                else:
                    print(f"[*] Motion extended recording deadline (+{userdata['video_timeout']}s)")
            else:
                # Image mode: pull a single frame in a background thread
                threading.Thread(
                    target=fetch_frame,
                    args=(userdata["camera_ip"], userdata["out_dir"]),
                    daemon=True,
                ).start()

        elif payload.lower() == "off":
            # In video mode "off" is informational — the recording runs until
            # max_duration seconds pass with no new "on" event.
            pass

    except Exception:
        print("[!] Exception in on_message:", file=sys.stderr)
        traceback.print_exc()


def on_callback_exception(client, userdata, callback, exception):
    print(f"[!] Exception in paho callback {callback.__name__}: {exception}", file=sys.stderr)
    traceback.print_exception(type(exception), exception, exception.__traceback__)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    device       = args.hostname if args.hostname else "+"
    motion_topic = f"{args.prefix}sensor/{device}/motion"
    ldr_topic    = f"{args.prefix}sensor/{device}/ldr"
    # cmd_topic requires a specific hostname (not wildcard)
    cmd_topic    = f"{args.prefix}sensor/{args.hostname}/cmd" if args.hostname else None

    if args.ldr_trigger and not args.hostname:
        print("[!] --ldr-trigger requires --hostname to target a specific device", file=sys.stderr)
        sys.exit(1)

    userdata = {
        "motion_topic":      motion_topic,
        "ldr_topic":         ldr_topic,
        "cmd_topic":         cmd_topic,
        "ldr_trigger":       args.ldr_trigger,
        "camera_ip":         args.camera,
        "out_dir":           args.outdir,
        "video_mode":        args.video,
        "video_timeout":     args.video_timeout,
        "recording_thread":  None,
        "stop_event":        None,
        "current_response":  None,
        "last_motion_time":  None,   # updated on every "on" event; drives the deadline
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
    print(f"[*] Fetching from http://{args.camera}/{'sustain?stream=1' if args.video else 'control?still=1'}")
    if args.video:
        print(f"[*] Video timeout: {args.video_timeout}s")
    if args.ldr_trigger:
        print(f"[*] LDR trigger on connect → {cmd_topic}")
    client.connect(args.broker, args.port, keepalive=60)

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n[*] Stopped.")
        if userdata["stop_event"]:
            userdata["stop_event"].set()
        client.disconnect()


if __name__ == "__main__":
    main()
