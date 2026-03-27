#!/usr/bin/env python3
"""
ESP32-CAM MQTT event receiver with HTTP frame pull or video recording,
with optional YOLO-based cat detection filtering.

Listens for motion events over MQTT, then either:
  - (default) fetches a single JPEG via GET /control?still=1
  - (--video)  records a video from the MJPEG stream at /sustain?stream=1,
               stopping --video-timeout seconds after the last motion event.

YOLO filtering (--yolo):
  Images: only saved if a target class is detected; boxes are drawn.
  Videos: decoded into per-frame JPEGs saved to {video}_frames/; frames
          without detections are discarded. The original .avi is kept.
  Inference runs in a separate process so MQTT handling is never blocked.
  Default model: yolo11n.pt (auto-downloaded by ultralytics on first run).
  Default target class: 15 (cat in COCO). Use --yolo-classes to change.

Also supports LDR (light sensor) readings for cat feeder food-level tracking:
  - Subscribes to {prefix}sensor/{hostname}/ldr and logs readings
  - Use --ldr-trigger to request a reading on startup (requires --hostname)

Usage:
    pip install paho-mqtt requests ultralytics opencv-python

    # Image mode, no YOLO:
    python3 mqtt_frame_receiver.py --broker 192.168.1.x --camera 192.168.1.y

    # Image mode with YOLO cat filter:
    python3 mqtt_frame_receiver.py --broker 192.168.1.x --camera 192.168.1.y --yolo

    # Video mode with YOLO:
    python3 mqtt_frame_receiver.py --broker 192.168.1.x --camera 192.168.1.y \\
        --video --yolo [--video-timeout 300]

    # Trigger an LDR reading on connect:
    python3 mqtt_frame_receiver.py --broker 192.168.1.x --camera 192.168.1.y \\
        --hostname esp32cam --ldr-trigger
"""

import argparse
import json
import multiprocessing
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
YOLO_QUEUE_MAXSIZE    = 100   # drop frames rather than grow unbounded
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="ESP32-CAM MQTT event receiver + HTTP frame/video puller")
    p.add_argument("--broker",        default=DEFAULT_BROKER,        help="MQTT broker IP/hostname")
    p.add_argument("--port",          default=DEFAULT_PORT, type=int, help="MQTT broker port (default 1883)")
    p.add_argument("--prefix",        default=DEFAULT_TOPIC_PREFIX,  help="mqtt_topic_prefix set in ESP32 config")
    p.add_argument("--hostname",      default="",                     help="ESP32 hostname (omit to accept any device)")
    p.add_argument("--camera",        required=True,                  help="ESP32 camera IP for HTTP requests")
    p.add_argument("--outdir",        default=DEFAULT_OUTPUT_DIR,     help="Directory to save files")
    p.add_argument("--user",          default="",                     help="MQTT broker username (if required)")
    p.add_argument("--password",      default="",                     help="MQTT broker password (if required)")
    p.add_argument("--video",         action="store_true",            help="Record video instead of single frame")
    p.add_argument("--video-timeout", default=DEFAULT_VIDEO_TIMEOUT, type=int, metavar="SECONDS",
                   help=f"Stop recording N seconds after last motion event (default {DEFAULT_VIDEO_TIMEOUT})")
    p.add_argument("--ldr-trigger",   action="store_true",
                   help="Send LDR measurement command on connect (requires --hostname)")
    # YOLO options
    p.add_argument("--yolo",          action="store_true",
                   help="Enable YOLO-based filtering (requires ultralytics)")
    p.add_argument("--yolo-model",    default="yolo11n.pt", metavar="MODEL",
                   help="YOLO model file (default: yolo11n.pt, auto-downloaded)")
    p.add_argument("--yolo-conf",     default=0.25, type=float, metavar="CONF",
                   help="Minimum detection confidence (default: 0.25)")
    p.add_argument("--yolo-classes",  default=[15], type=int, nargs="+", metavar="ID",
                   help="COCO class IDs to keep (default: 15=cat). Use 0 for person, etc.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# YOLO worker — runs in a separate process
# ---------------------------------------------------------------------------

def yolo_worker(queue, model_path, conf_threshold, target_classes):
    """Pull work items from queue and run YOLO inference.

    Items are tuples:
      ("image", jpeg_bytes, out_dir, timestamp_str)
      ("video", video_path, out_dir)
      None  → shutdown sentinel
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[yolo] ultralytics not installed. Run: pip install ultralytics", file=sys.stderr)
        return

    print(f"[yolo] Loading {model_path} ...", flush=True)
    model = YOLO(model_path)
    names = model.names
    print(f"[yolo] Ready. Target classes: "
          f"{[names.get(c, str(c)) for c in target_classes]}  conf≥{conf_threshold}",
          flush=True)

    while True:
        item = queue.get()
        if item is None:
            print("[yolo] Shutting down.", flush=True)
            break
        kind = item[0]
        try:
            if kind == "image":
                _, jpeg_bytes, out_dir, ts = item
                _yolo_image(model, jpeg_bytes, out_dir, ts, conf_threshold, target_classes)
            elif kind == "video":
                _, video_path, out_dir = item
                _yolo_video(model, video_path, out_dir, conf_threshold, target_classes)
        except Exception:
            print(f"[yolo] Error processing {kind}:", file=sys.stderr)
            traceback.print_exc()


def _draw_boxes(frame, hits, names):
    import cv2
    for b in hits:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        label = f"{names[int(b.cls)]} {float(b.conf):.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(y1 - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def _yolo_image(model, jpeg_bytes, out_dir, ts, conf, target_classes):
    """Run inference on a JPEG; save annotated image only if target detected."""
    import cv2
    import numpy as np

    arr   = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        print("[yolo] Could not decode image", file=sys.stderr)
        return

    results = model(frame, conf=conf, verbose=False)[0]
    hits = [b for b in results.boxes if int(b.cls) in target_classes]
    if not hits:
        return  # catless frame — discard

    best_conf = max(float(b.conf) for b in hits)
    _draw_boxes(frame, hits, model.names)

    filename = os.path.join(out_dir, f"cat_{ts}_{best_conf:.2f}.jpg")
    cv2.imwrite(filename, frame)
    print(f"[yolo] {filename}  ({len(hits)} detection(s), conf {best_conf:.2f})", flush=True)


def _yolo_video(model, video_path, out_dir, conf, target_classes):
    """Process a recorded video: save only frames with target detections.

    Output goes to {out_dir}/{video_basename}_frames/frame_NNNNN.jpg.
    The original video file is kept. The frames folder is removed if empty.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[yolo] Cannot open {video_path}", file=sys.stderr)
        return

    base       = os.path.splitext(os.path.basename(video_path))[0]
    frames_dir = os.path.join(out_dir, f"{base}_frames")
    os.makedirs(frames_dir, exist_ok=True)

    total = saved = 0
    t0 = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            total += 1

            results = model(frame, conf=conf, verbose=False)[0]
            hits = [b for b in results.boxes if int(b.cls) in target_classes]
            if not hits:
                continue

            _draw_boxes(frame, hits, model.names)
            cv2.imwrite(os.path.join(frames_dir, f"frame_{total:05d}.jpg"), frame)
            saved += 1
    finally:
        cap.release()

    elapsed = time.time() - t0
    if saved == 0:
        os.rmdir(frames_dir)
        print(f"[yolo] {os.path.basename(video_path)}: "
              f"no detections in {total} frames ({elapsed:.1f}s) — frames folder removed",
              flush=True)
    else:
        print(f"[yolo] {os.path.basename(video_path)}: "
              f"{saved}/{total} frames with detections → {frames_dir}/  ({elapsed:.1f}s)",
              flush=True)


# ---------------------------------------------------------------------------
# Image mode
# ---------------------------------------------------------------------------

def fetch_frame(camera_ip, out_dir, yolo_queue=None):
    """Fetch a single JPEG from /control?still=1.

    If yolo_queue is set the bytes are queued for YOLO filtering and the raw
    file is not saved directly (YOLO worker saves only if cat detected).
    Without a queue the frame is saved immediately as motion_TIMESTAMP.jpg.
    """
    url = f"http://{camera_ip}/control?still=1"
    try:
        t0   = time.time()
        resp = requests.get(url, timeout=HTTP_TIMEOUT)
        elapsed = time.time() - t0
        if resp.status_code == 200 and resp.content:
            ts = time.strftime("%Y%m%d_%H%M%S")
            if yolo_queue is not None:
                try:
                    yolo_queue.put_nowait(("image", resp.content, out_dir, ts))
                    print(f"[+] Frame queued for YOLO  ({len(resp.content)} bytes, {elapsed:.2f}s)",
                          flush=True)
                except Exception:
                    print("[!] YOLO queue full — frame dropped", file=sys.stderr)
            else:
                filename = os.path.join(out_dir, f"motion_{ts}.jpg")
                with open(filename, "wb") as f:
                    f.write(resp.content)
                print(f"[+] Frame saved: {filename}  ({len(resp.content)} bytes, {elapsed:.2f}s)")
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
            cl_idx = buf.find(b"Content-Length:")
            if cl_idx < 0:
                break
            eol = buf.find(b"\r\n", cl_idx)
            if eol < 0:
                break
            try:
                frame_len = int(buf[cl_idx + 15 : eol].strip())
            except ValueError:
                buf = buf[cl_idx + 15:]
                continue
            hdr_end = buf.find(b"\r\n\r\n", cl_idx)
            if hdr_end < 0:
                break
            data_start = hdr_end + 4
            data_end   = data_start + frame_len
            if len(buf) < data_end:
                break
            yield buf[data_start : data_end]
            buf = buf[data_end:]


def record_video(camera_ip, out_dir, stop_event, max_duration, userdata, yolo_queue=None):
    """Stream MJPEG from /sustain?stream=1 and write to a video file.

    The recording deadline is based on the *last* motion "on" event stored in
    userdata["last_motion_time"], so every new motion event extends the window
    by max_duration seconds from that moment.  "off" events are ignored here.
    Reconnects automatically if the stream stalls.

    If yolo_queue is set the finished video file path is queued for YOLO
    processing (frames with detections extracted; original file kept).
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("[!] opencv-python required for video mode.  Run: pip install opencv-python",
              file=sys.stderr)
        return

    url      = f"http://{camera_ip}/sustain?stream=1"
    ts       = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(out_dir, f"video_{ts}.avi")
    print(f"[*] Recording video → {filename}  (idle-stop after {max_duration}s of no motion)")

    writer      = None
    frame_count = 0
    t0          = time.time()
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
                    arr   = np.frombuffer(jpg, dtype=np.uint8)
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
                        last      = userdata.get("last_motion_time") or t0
                        remaining = max(0, max_duration - (now - last))
                        fps       = frame_count / (now - t0)
                        print(f"[*] {frame_count} frames  {fps:.1f} fps  "
                              f"deadline in {remaining:.0f}s", flush=True)
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
            if yolo_queue is not None:
                try:
                    yolo_queue.put_nowait(("video", filename, out_dir))
                    print(f"[*] Video queued for YOLO processing", flush=True)
                except Exception:
                    print("[!] YOLO queue full — video not queued", file=sys.stderr)
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
        yolo = " + YOLO" if userdata["yolo_queue"] is not None else ""
        print(f"[+] Connected ({mode}{yolo} mode). Subscribed to {motion_topic}")
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
                data         = json.loads(payload)
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
            yolo_queue = userdata.get("yolo_queue")
            if userdata["video_mode"]:
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
                            yolo_queue,
                        ),
                        daemon=True,
                    )
                    userdata["recording_thread"] = t
                    t.start()
                else:
                    print(f"[*] Motion extended recording deadline (+{userdata['video_timeout']}s)")
            else:
                threading.Thread(
                    target=fetch_frame,
                    args=(userdata["camera_ip"], userdata["out_dir"], yolo_queue),
                    daemon=True,
                ).start()

        elif payload.lower() == "off":
            # In video mode "off" is informational — recording runs until deadline.
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
    cmd_topic    = f"{args.prefix}sensor/{args.hostname}/cmd" if args.hostname else None

    if args.ldr_trigger and not args.hostname:
        print("[!] --ldr-trigger requires --hostname", file=sys.stderr)
        sys.exit(1)

    # Start YOLO worker process if requested
    yolo_queue   = None
    yolo_process = None
    if args.yolo:
        yolo_queue = multiprocessing.Queue(maxsize=YOLO_QUEUE_MAXSIZE)
        yolo_process = multiprocessing.Process(
            target=yolo_worker,
            args=(yolo_queue, args.yolo_model, args.yolo_conf, args.yolo_classes),
            daemon=True,
        )
        yolo_process.start()
        print(f"[*] YOLO worker started  model={args.yolo_model}  "
              f"conf={args.yolo_conf}  classes={args.yolo_classes}")

    userdata = {
        "motion_topic":     motion_topic,
        "ldr_topic":        ldr_topic,
        "cmd_topic":        cmd_topic,
        "ldr_trigger":      args.ldr_trigger,
        "camera_ip":        args.camera,
        "out_dir":          args.outdir,
        "video_mode":       args.video,
        "video_timeout":    args.video_timeout,
        "yolo_queue":       yolo_queue,
        "recording_thread": None,
        "stop_event":       None,
        "current_response": None,
        "last_motion_time": None,
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
    print(f"[*] Camera: http://{args.camera}/{'sustain?stream=1' if args.video else 'control?still=1'}")
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
    finally:
        if yolo_queue is not None:
            yolo_queue.put(None)       # shutdown sentinel
        if yolo_process is not None:
            yolo_process.join(timeout=60)
            if yolo_process.is_alive():
                yolo_process.terminate()


if __name__ == "__main__":
    main()
